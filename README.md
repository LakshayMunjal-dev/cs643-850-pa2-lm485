# Programming Assignment 2 - ML Model with Docker and AWS

This repository contains the implementation of a machine learning model using **Spark**, **Docker**, and **AWS** infrastructure for distributed training and prediction. The project demonstrates how to set up a scalable environment, deploy models, and containerize the application using Docker.

# Github Repo Link

   [Link to Github Repository](https://github.com/LakshayMunjal-dev/cs643-850-pa2-lm485)

### Table of Contents

1. [Overview](#overview)
2. [Docker Setup](#docker-setup)
3. [AWS Setup](#aws-setup)
4. [Model Training](#model-training)
5. [Running the Model](#running-the-model)
6. [Push Docker Image to Docker Hub](#push-docker-image-to-docker-hub)
7. [File Structure](#file-structure)

## Overview

This project focuses on training and running machine learning models on a **distributed Spark cluster** using **AWS Elastic MapReduce (EMR)**. The workflow includes setting up an EMR cluster, uploading datasets to **S3**, containerizing the application with **Docker**, and executing both training and prediction tasks.

## Docker Setup

The project utilizes **Docker** to package the environment, dependencies, and Python scripts, ensuring consistent deployment across systems. Docker enables seamless running of the machine learning models in a controlled environment.

### Steps for Docker Setup:

1. **Install Docker**: 
   - Ensure Docker is installed on your machine. If it’s not installed, follow the [Docker installation guide](https://docs.docker.com/get-docker/).

2. **Build the Docker Image**:
   - Use the following command to build the Docker image from the Dockerfile:
   ```bash
   docker build -t lm485/programmingassignment2 .
   ```

3. **Run the Docker Container**:
   - Once the image is built, run the container with:
   ```bash
   docker run lm485/programmingassignment2
   ```

4. **Push the Docker Image**:
   - After verifying that the container works as expected, push the image to Docker Hub:
   ```bash
   docker push lm485/programmingassignment2
   ```

   [Link to Docker Hub Repository](https://hub.docker.com/repository/docker/lm485/programmingassignment2/general)

## AWS Setup

To leverage **AWS** for distributed model training, the following setup is required:

### EMR Cluster Setup:

1. **Launch an EMR Cluster**:
   - Create an EMR cluster with the following configuration:
     - **EMR Version**: 5.36.0
     - **Hadoop**: 2.10.1, **Spark**: 2.4.8, **Zeppelin**: 0.10.0
     - **EC2 Instance Type**: `m5.xlarge` for primary, core, and task instances.
     - **Number of Instances**: 4 instances for core and task.

2. **Set Up Security**:
   - In the security group settings, configure inbound rules to allow SSH access from your IP.

3. **Create an S3 Bucket**:
   - Create an S3 bucket to upload the datasets and Python scripts (training and prediction).
   - After training the model, the model file will be saved in this bucket for future predictions.

### Sync Files to EMR Cluster:
To sync files from the S3 bucket to the EMR cluster, use this command:
```bash
aws s3 sync s3://your-bucket-name/ .
```

## Model Training

Training the model occurs on the EMR cluster using Spark. Follow these steps to train the model:

1. **SSH into the EMR Cluster**:
   Use SSH to connect to the master node of your EMR cluster:
   ```bash
   ssh -i "your-key.pem" hadoop@your-ec2-public-ip
   ```

2. **Submit the Training Job**:
   Once logged into the master node, run the training script using Spark:
   ```bash
   spark-submit /opt/lm485-wine-train.py
   ```

3. **Monitor the Training Process**:
   Monitor the training job in the EMR UI. The training progress and logs will be available there. Once complete, the trained model will be saved in your S3 bucket.

## Running the Model

After the model has been trained, use the following steps to run predictions on the validation dataset.

1. **Submit the Prediction Job**:
   Run the prediction script using Spark:
   ```bash
   spark-submit scripts/lm485-wine-predict.py
   ```

   This will load the trained model from S3 and apply it to the validation dataset for predictions.

## Push Docker Image to Docker Hub

For replicable and sharable environments, the Docker image containing the necessary dependencies and code is pushed to Docker Hub.

1. **Log in to Docker**:
   To authenticate with Docker Hub:
   ```bash
   docker login
   ```

2. **Push the Docker Image**:
   Push the image to your Docker Hub repository:
   ```bash
   docker push lm485/programmingassignment2
   ```

   [Link to Docker Hub Repository](https://hub.docker.com/repository/docker/lm485/programmingassignment2/general)

## File Structure

The directory structure for this project is as follows:

```
.
├── Dataset/
├── Dockerfile
├── requirements.txt
├── scripts/
│   ├── lm485-wine-predict.py
│   └── lm485-wine-train.py
└── README.md
```

- `Dataset/`: Contains the training and validation datasets.
- `Dockerfile`: The Dockerfile used to build the Docker image.
- `scripts/`: Python scripts for training (`lm485-wine-train.py`) and prediction (`lm485-wine-predict.py`).
- `requirements.txt`: Python dependencies for the project.
- `README.md`: This documentation file.