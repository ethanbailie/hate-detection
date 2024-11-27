import sagemaker
from sagemaker.huggingface import HuggingFace
import os
import dotenv

dotenv.load_dotenv()

## initialize sagemaker session
sagemaker_session = sagemaker.Session()

## define iam role with sagemaker permissions
role = os.getenv('ARN')

## define hyperparameters
hyperparameters = {
    'epochs': 3,
    'batch_size': 16,
    'learning_rate': 2e-5
}

## define huggingface estimator with output_path set to your s3 bucket
huggingface_estimator = HuggingFace(
    entry_point='training.py',                    
    source_dir='.',           
    instance_type='ml.p3.2xlarge',                
    instance_count=1,
    role=role,
    transformers_version='4.46.3',                    
    pytorch_version='2.5.1',                         
    py_version='py310',                             
    hyperparameters=hyperparameters,
    output_path='s3://hate-detection'             
)

## define the s3 path to your training data
training_data_uri = 's3://hate-detection/data'

## start the training job by specifying the input data location
huggingface_estimator.fit({'train': training_data_uri})