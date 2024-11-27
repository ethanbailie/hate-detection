# Hate Speech Detection with Fine-Tuned BERT

This repository contains code for fine-tuning a BERT model to detect hate speech. The project supports both local training and deployment on AWS SageMaker for scalable model training and inference.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Local Training](#local-training)
  - [SageMaker Training](#sagemaker-training)
- [Environment Setup for SageMaker](#environment-setup-for-sagemaker)
- [Repository Structure](#repository-structure)
- [License](#license)

## Overview

This project leverages a fine-tuned BERT model for hate speech detection. Users can train the model locally or deploy it to AWS SageMaker for large-scale training.

## Features

- **Local Training:** Train the model using local resources with customizable epochs.
- **SageMaker Deployment:** Deploy the training process to AWS SageMaker for scalability and efficiency.
- **Preconfigured Scripts:** Includes scripts for data preprocessing, model training, and SageMaker deployment.

## Requirements

Before getting started, ensure you have the following:

- Python 3.6 (for SageMaker compatibility)
- Conda installed on your system
- AWS CLI configured with your credentials
- SageMaker permissions and a valid ARN
- An S3 bucket for storing training data and model artifacts
- Required Python dependencies (listed in `requirements.txt`)

## Installation

### Clone this repository:

```bash
git clone https://github.com/ethanbailie/hate-detection.git
cd hate-detection
```

### Clone this repository:

```bash
git clone https://github.com/ethanbailie/hate-detection.git
cd hate-detection
```

### Install dependancies
```bash
pip install -r requirements.txt
```

## Usage

### Local Training
To train the model locally, execute the following command (adjust --epochs as needed):
```bash
python training.py --data_dir ./data --model_dir ./model_output --epochs 1
```

This command:
- Loads training data from the ./data directory.
- Saves the trained model to the ./model_output directory.
- Runs for the specified number of epochs.

### SageMaker Training
For training on SageMaker, use the sm_estimator.py script.

Ensure you:
- Set up your AWS environment variables (Having AWS CLI properly installed on your device is good enough).
- Provide the necessary configurations (see Environment Setup for SageMaker).

## Environment Setup for SageMaker
### ARN Configuration
Add your ARN to the .env in the repo (this should be a role with permissions to GetObject, PutObject, and access to SageMaker operations):
```
SAGEMAKER_ROLE_ARN=your_sagemaker_role_arn
```

### S3 Bucket
Update the sm_estimator.py script with the correct S3 bucket name for storing data:
```
s3_bucket = 'your-s3-bucket-name'
```

## Repository Structure
```bash
hate-detection/
├── data/                # Training and validation datasets
├── model_output/        # Directory to save trained model locally
├── raw_data_intake.py   # Script for processing raw dataset
├── tokenization.py      # Script for processing the message text into tokens
├── training.py          # Script for local training
├── sm_estimator.py      # Script for SageMaker training
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── .env                 # Environment variable file for sensitive data
```

## License

This project is licensed under the MIT License.
