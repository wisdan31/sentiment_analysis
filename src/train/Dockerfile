# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to the project root
WORKDIR /app

# Copy only the necessary files and folders
COPY requirements.txt .
COPY src /app/src

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory to train and run the training script
WORKDIR /app/src/train
CMD ["python", "train.py"]
