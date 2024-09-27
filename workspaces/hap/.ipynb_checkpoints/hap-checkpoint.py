# home auto pilot
import logging

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class HomeAutoPilot(object):
    def __init__(self, input_size = 9, hidden_size = 128, num_lstm_layers = 2, num_device_actions = 2, num_devices = 213):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.model = MultiTaskLSTM(input_size=input_size, num_lstm_layers=num_lstm_layers, hidden_size=hidden_size, num_device_actions=num_device_actions, num_devices=num_devices)
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f'Using device: {self.device}')
        self.model.to(self.device)
        self.__loaded_or_trained = False
    
    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()
        self.__loaded_or_trained = True
    
    def train_model(self, input_tensors, act_dev_tensors, act_state_tensors, act_timing_tensors, train_epoch = 32, batch_size = 32, model_path = None):
        # setup the loss functions and optimizer
        criterion_timing = nn.MSELoss()  # Loss for timing prediction (regression)
        criterion_action = nn.CrossEntropyLoss()  # Loss for action type prediction (classification)
        criterion_device = nn.CrossEntropyLoss()  # Loss for device prediction (classification)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # divide the dataset into train/val/test sets
        total_size = len(input_tensors)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size

        # create pytorch datasets
        dataset = SmartHomeDataset(input_tensors, act_dev_tensors, act_state_tensors, act_timing_tensors)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

        # train the model with the prepped data
        self.__train(train_loader, criterion_action, criterion_timing, criterion_device, optimizer, num_epochs = train_epoch)
        self.__test(test_loader, criterion_action, criterion_timing, criterion_device)
        # save model if provided a path
        if model_path: torch.save(self.model.state_dict(), model_path)
        
        # tell everybody we are ready
        self.__loaded_or_trained = True

    def eval(self, input_tensor):
        '''
        evaluate a single tensor input
        '''
        self.model.eval()
        if not self.__loaded_or_trained: raise ValueError('model not intialized yet, please load or train a new model')
        input_length = torch.tensor([len(input_tensor)])
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        # no need to pad here since it's not a batch
        # input_tensor = pad_sequence(input_tensor, batch_first=True, padding_value=0)
        with torch.no_grad():
            time_pred, action_pred, device_pred = self.model(input_tensor, input_length)
        
        predicted_device_idx = torch.argmax(device_pred).item()
        predicted_action_idx = torch.argmax(action_pred).item()
        predicted_timing = time_pred.item()
        return predicted_timing, predicted_device_idx, predicted_action_idx
    
    def __train(self, train_data_loader, criterion_action, criterion_timing, criterion_device, optimizer, num_epochs=3):
        for epoch in range(num_epochs):
            self.model.train()
        
            for batch_idx, (inputs, y_act_device, y_act_state, y_timing, lengths) in tqdm(enumerate(train_data_loader)):
                inputs, y_device, y_action, y_timing, lengths = (
                    inputs.float().to(self.device), 
                    y_act_device.long().to(self.device), 
                    y_act_state.long().to(self.device), 
                    y_timing.float().to(self.device), 
                    lengths # this guy stays in CPU
                )
                
                # Forward pass
                timing_pred, action_pred, device_pred = self.model(inputs, lengths)
            
                # Calculate losses
                action_loss = criterion_action(action_pred, y_action)
                timing_loss = criterion_timing(timing_pred, y_timing)
                device_loss = criterion_device(device_pred, y_device)
    
                # TODO: parameterize
                device_weight = 1.0  # Higher weight for more importance
                action_weight = 1.0  # Higher weight for more importance
                timing_weight = 0.5  # Lower weight for less importance
                
                # Calculate the weighted loss
                weighted_loss = (device_weight * device_loss) + (action_weight * action_loss) + (timing_weight * timing_loss)
    
                # Backpropagation and optimization
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()
                
            self.logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {weighted_loss.item():.4f}')
    
    def __test(self, test_loader, criterion_action, criterion_timing, criterion_device):
        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0.0
        correct_device = 0
        correct_action = 0
        total_samples = 0
        
        with torch.no_grad():  # No need to calculate gradients during testing
            for inputs, y_device, y_action, y_timing, lengths in test_loader:
                # Move data to the same device as the model
                inputs = inputs.float().to(self.device)
                y_device = y_device.long().to(self.device)
                y_action = y_action.long().to(self.device)
                y_timing = y_timing.float().to(self.device)
                lengths = lengths.to(self.device)
        
                # Forward pass
                timing_pred, action_pred, device_pred = self.model(inputs, lengths)
        
                # Squeeze the predictions if necessary
                timing_pred = timing_pred.squeeze(1)
                action_pred = action_pred.squeeze(1)
                device_pred = device_pred.squeeze(1)
                
                # Calculate loss
                loss_timing = criterion_timing(timing_pred, y_timing)
                loss_action = criterion_action(action_pred, y_action)
                loss_device = criterion_device(device_pred, y_device)
        
                # Accumulate total loss
                test_loss += (loss_timing + loss_action + loss_device).item()
        
                # Calculate accuracy for device and action predictions
                _, predicted_device = torch.max(device_pred, 1)
                _, predicted_action = torch.max(action_pred, 1)
                
                correct_device += (predicted_device == y_device).sum().item()
                correct_action += (predicted_action == y_action).sum().item()
                total_samples += y_device.size(0)
        
        # Calculate average loss
        test_loss /= len(test_loader)
        
        # Calculate accuracy
        device_accuracy = correct_device / total_samples
        action_accuracy = correct_action / total_samples
        
        self.logger.info(f'Test Loss: {test_loss:.4f}')
        self.logger.info(f'Device Prediction Accuracy: {device_accuracy:.4f}')
        self.logger.info(f'Action Prediction Accuracy: {action_accuracy:.4f}')

class MultiTaskLSTM(nn.Module):
    def __init__(self, input_size, num_device_actions, num_devices, hidden_size=128, num_lstm_layers=4):
        super(MultiTaskLSTM, self).__init__()
        # LSTM shared layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_lstm_layers, 
            batch_first=True, 
            dropout=0.2)
        
        # Timing prediction head (regression)
        self.timing_pred_head = nn.Linear(hidden_size, 1)  # 1 output for time prediction
        
        # Action classification head (classification)
        self.act_state_head = nn.Linear(hidden_size, num_device_actions)  # num_device_actions output for classification
        
        # Device classification head (classification)
        self.act_device_head = nn.Linear(hidden_size, num_devices)  # num_devices output for device classification
    
    def forward(self, x, lengths):
        # Process the padded sequence with LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = lstm_out[range(len(lstm_out)), lengths - 1]
        
        # Timing and action predictions
        timing_pred = self.timing_pred_head(lstm_out)
        action_pred = self.act_state_head(lstm_out)
        device_pred = self.act_device_head(lstm_out)
        
        return timing_pred, action_pred, device_pred


class SmartHomeDataset(Dataset):
    def __init__(self, input_tensors, act_dev_tensors, act_state_tensors, act_timing_tensors):
        self.input_tensors = input_tensors
        self.act_dev_tensors = act_dev_tensors
        self.act_state_tensors = act_state_tensors
        self.act_timing_tensors = act_timing_tensors

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return (self.input_tensors[idx], 
                self.act_dev_tensors[idx], 
                self.act_state_tensors[idx], 
                self.act_timing_tensors[idx])

def _collate_fn(batch):
    # Separate inputs and targets
    inputs, y_device, y_action, y_timing = zip(*batch)
    
    # Pad sequences for inputs (batch_first=True makes it [batch_size, seq_len, features])
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    
    # Convert targets to tensors (they should all have the same length as they're scalar values)
    y_device = torch.stack(y_device)
    y_action = torch.stack(y_action)
    y_timing = torch.stack(y_timing)
    
    # Compute lengths for each sequence (before padding)
    lengths = torch.tensor([len(seq) for seq in inputs])
    
    return inputs_padded, y_device, y_action, y_timing, lengths