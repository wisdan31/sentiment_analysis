# Train

`docker build -f ./src/train/Dockerfile -t sentiment-analysis .` — build image

`docker run --rm -v ${PWD}/outputs/models:/app/outputs/models sentiment-analysis` — run container. Saved at `outputs/models`
