
# What is this
I wanna try to model the action prediction such that, given the current state of the home, and the previous sensors/actions happened in the past timewindow `T`, what's likely the next action (e.g., control the kitchen ceiling light) will happen (e.g., turn-off)? But I do not have enough data from Home Assistant to train a sufficient model, so I am finding a way to create a dataset that would be close to the real-world home assistant dataset. 

This repo is an experiment (contains raw JuypterNotebook pages) and trying to demonstrate the idea of using the [CASAS](https://casas.wsu.edu/datasets/) dataset to form a suitable dataset and test the effectiveness of a simple transformer model.

# What is the dataset?

The CASAS dataset is a collection of sensor data and tagged activities of an actual smart home and real human activities.

Sensors: (timestamp, sensor_id, value, activity tagging)
```csv
2009-06-10 00:00:00.024668	T003	19
2009-06-10 00:00:46.069471	T005	18.5
2009-06-10 00:00:47.047655	T003	18.5
2009-06-10 00:01:17.070215	T005	18
2009-06-10 00:04:38.092963	T005	18.5
...
2009-06-10 03:22:18.063157	M011	OFF
2009-06-10 03:22:18.078709	M012	OFF
2009-06-10 03:22:19.001209	M022	ON
2009-06-10 03:22:24.015902	M022	OFF
2009-06-10 03:25:19.059284	M012	ON
2009-06-10 03:25:19.086432	M011	ON
2009-06-10 03:25:24.054674	M011	OFF
2009-06-10 03:25:24.070558	M012	OFF	Night wandering end
```

Activities:
```
Bed to toilet (30)
Breakfast (48)
R1 sleep (50)
R1 wake (53)
R1 work in office (46)
Dinner (42)
Laundry (10)
Leave home (69)
Lunch (37)
Night wandering (67)
R2 sleep (52)
R2 take medicine (44)
R2 wake (52)
```

# Detail model development architecture
* [Fullframe-Multihead-Transformer model for action prediction](doc/Model-exploration_full-frame-multi-head-transformer.md)

# Dev envrionment setup
## Create the Pytorch environment

```sh
docker compose up -d
```

## Get the Jupyter lab url

```sh
docker logs pytorch-gpu-dev | grep http://127.0.0.1:8889
```
visit the url locally

## FAQ

* How to access notebooks locally

> `{docker-compose-root}/workspaces`