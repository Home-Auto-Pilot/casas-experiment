import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

def read_and_clean(fname):
    df = pd.read_csv(fname)
    df = df[df['state'] != 'unavailable']
    df.rename(columns={'last_changed': 'datetime'}, inplace = True)
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%S.%fZ', utc=True)
    df['second'] = df['datetime'].dt.second
    df['minute'] = df['datetime'].dt.minute
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.day
    df['weekofmonth'] = (df['datetime'].dt.day - 1) // 7 + 1
    df['monthofyear'] = df['datetime'].dt.month
    df_switches = df[df['state'].isin({'on', 'off'})]
    df_sensors = df[df['entity_id'].str.contains("sensor")]
    df_sensors = df_sensors[df_sensors['entity_id'].str.contains('status|last_seen') == False]
    return df_switches, df_sensors

def combine_data(data_files):
    df_switches_list = []
    df_sensors_list = []
    for data in [read_and_clean(fname) for fname in data_files]:
        df_switches_list.append(data[0])
        df_sensors_list.append(data[1])
    df_sensors_combined = pd.concat(df_sensors_list)
    df_sensors_combined['type'] = 0 # sensor
    # correctly factorize the mixed states (numerical, timestamp and categorial states)
    categorical_state_mask = pd.to_numeric(df_sensors_combined['state'], errors='coerce').isnull()
    numerical_state_mask = ~categorical_state_mask
    df_sensors_combined.loc[categorical_state_mask, 'state'], unique_sensor_states = pd.factorize(df_sensors_combined['state'][categorical_state_mask])
    df_sensors_combined.loc[numerical_state_mask, 'state'] = pd.to_numeric(df_sensors_combined['state'][numerical_state_mask])
    df_switches_combined = pd.concat(df_switches_list)
    df_switches_combined['type'] = 1 # event
    df_switches_combined['state'], unique_event_states = pd.factorize(df_switches_combined['state'])
    df_combined = pd.concat([df_switches_combined, df_sensors_combined])
    df_combined['entity_id'], unique_entity_ids = pd.factorize(df_combined['entity_id'])
    df_combined = df_combined.sort_values(by='datetime')
    return df_combined, df_sensors_combined, df_switches_combined, unique_sensor_states, unique_entity_ids, unique_event_states

def build_sequence(data_files):
    df_combined, df_sensors_combined, df_switches_combined, unique_sensor_states, unique_entity_ids, unique_event_states = combine_data(data_files)
    timestamps = df_combined['datetime'].values
    data = df_combined.drop('datetime', axis=1).values 
    type_loc = df_combined.drop('datetime', axis=1).columns.get_loc('type')
    entity_id_loc = df_combined.drop('datetime', axis=1).columns.get_loc('entity_id')
    state_loc = df_combined.drop('datetime', axis=1).columns.get_loc('state')
    # Parameters
    window_size_seconds = 300  # 5 mins = 86400 seconds
    step_size = 1  # 1-second step size
    
    # Prepare to store windows
    input_sequences = []
    action_device_sequences = []
    action_state_sequences = []
    action_timing_sequences = []
    for i in tqdm(range(len(timestamps))):
        # Define the end time for the current window
        end_time = timestamps[i] + pd.Timedelta(seconds=window_size_seconds)
    
        # Collect data for the current window
        window_data = []
        last_pos = i
        for j in range(i, len(timestamps)):
            if timestamps[j] <= end_time:
                last_pos = j
                window_data.append(data[j].astype('float'))
            else:
                break  # Stop if we exceed the window time
        
        # Append to the sequences if the window is not empty
        # minimum number of snesory/action data needed is 10
        if len(window_data) < 10 or window_data[-1][type_loc] != 1: continue
        input_seq = window_data[:-1]
        input_sequences.append(input_seq)
        action_device_sequences.append(window_data[-1][entity_id_loc])
        action_state_sequences.append(window_data[-1][state_loc])
        timing = (timestamps[last_pos] - timestamps[last_pos-2]).astype('timedelta64[s]').astype('int64')
        action_timing_sequences.append(timing)
    return (input_sequences, action_device_sequences, action_state_sequences, action_timing_sequences, unique_sensor_states, unique_entity_ids, unique_event_states)

def prep_tensors(data_files, input_tr_path = 'input_seq_tensors.pt', dev_tr_path = 'act_dev_tensors.pt', state_tr_path = 'act_state_tensors.pt', timing_tr_path = 'act_timing_tensors.pt', device_id_map_path = 'device_id_map.pt', device_state_map_path = 'device_state_map.pt'):
    # extract / transform data
    input_sequences, action_device_sequences, action_state_sequences, action_timing_sequences, unique_sensor_states, unique_entity_ids, unique_event_states = build_sequence(data_files)
    #  tensorize
    input_tensors = [torch.tensor(input_sequence).to(torch.float32) for input_sequence in input_sequences]
    act_dev_tensors = [torch.tensor(dev).to(torch.float32) for dev in action_device_sequences]
    act_state_tensors = [torch.tensor(state).to(torch.float32) for state in action_state_sequences]
    act_timing_tensors = [torch.tensor(timing).to(torch.float32) for timing in action_timing_sequences]
    actionable_devices = set([act for act in action_device_sequences])
    # some stats
    n_act_dev = len(unique_entity_ids)
    actionable_states = unique_event_states
    n_act_state = len(actionable_states)
    print(f'number of unique actionalble devices: {n_act_dev}, number of actionable state: {n_act_state}')
    n_features = len(input_sequences[0][0])
    print(f'number of features: {n_features}')
    # save it?
    torch.save(input_tensors, input_tr_path)
    torch.save(act_dev_tensors, dev_tr_path)
    torch.save(act_state_tensors, state_tr_path)
    torch.save(act_timing_tensors, timing_tr_path)
    torch.save(unique_entity_ids, device_id_map_path)
    torch.save(unique_event_states, device_state_map_path)
    return (input_tensors, act_dev_tensors, act_state_tensors, act_timing_tensors, n_features, n_act_state, n_act_dev, unique_entity_ids, unique_event_states)