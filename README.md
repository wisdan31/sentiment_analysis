# Prerequisites

## Forking and Cloning from GitHub

To start using this project, you first need to get a local copy of this project. U can fork this project to your account, using "Fork" button at the top right part of the projects page, and then get the local copy of this project using `git clone`. After that, you should have this project on your machine with your work directory being project root.

**Note, that all commands provided are for Windows. They can be different on Linux and Mac**

## Docker

This project can be run in two ways: by running all related scripts using IDE or other instrument that allows to run python scripts, or by building Docker images and running Docker containers later on. To run using Docker, you need to install ([Docker Desktop](https://www.docker.com/products/docker-desktop)) — it is available for both Windows and Mac. After downloading the installer, run it, and follow the on-screen instructions. Once the installation is completed, you can open Docker Desktop to confirm it's running correctly. 

# Running project

Each part of project is logging it's actions in the terminal. You can check it to see if script finished or something went wrong.**Note, that model metrics will also appear in logs**

## Setup & Installation

If you will use Docker, you can skip this step. If you will run scripts, directly, before running the project, ensure that all required dependencies are installed. You can do this by running:

```bash
pip install --no-cache-dir -r requirements.txt
```

## Data loading

Firstly, you need to get data to train your model. Run:

```bash
python src/data_loader.py 
```

Or imply go to `src/` folder and run `data_loader.py` script in your IDE — after finishing, new folder named `data` will be created in `src` folder, and all necessary data will be downloaded from cloud.

## Train

To train the model using Docker: 

- Build the training Docker image:
```bash
docker build -f ./src/train/Dockerfile -t sentiment-analysis .  
```
- Then, run the container to train your model. After finishing, container will automatically remove itself and you should have model on your local machine in `outputs/models` folder:
```bash
docker run --rm `
  -v ${PWD}/outputs/models:/app/outputs/models `
  -v ${PWD}/src/data:/app/src/data `
  sentiment-analysis
```

Alternatively, the `train.py` script can also be run locally as follows:

```bash
python src/train/train.py
```

## Inference

Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

- Build the inference Docker image:
```bash
docker build -f ./src/inference/Dockerfile -t sentiment-inference . 
```

- Then, run the container to inference on new, unseen data . After finishing, container will automatically remove itself and you should have inference results on your local machine in `outputs/results` folder:
```bash
docker run --rm -v ${PWD}/outputs:/app/outputs sentiment-inference
```

Alternatively, the `inference.py` script can also be run locally as follows:

```bash
python src/train/inference.py
```

# Using Your Own Data for Inference  
By default, the inference script uses test.csv from the data folder, which is loaded by `data_loader.py`.

As the project is still under development and pretty clunky, there is currently no built-in method to input custom files.
As a temporarly bypass, to use your own dataset, manually replace `test.csv` in the data folder with your own file (renamed to `test.csv`) **before** running train script.

Ensure that your file follows the same structure as the original `test.csv` to prevent errors.

