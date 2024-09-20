# Create the Pytorch environment

```sh
docker compose up -d
```

# Get the Jupyter lab url

```sh
docker logs pytorch-gpu-dev | grep http://127.0.0.1:8889
```
visit the url locally

## FAQ

* How to access notebooks locally

> `{docker-compose-root}/workspaces`