# hate-detection
using finetuned BERT to detect hate speech

execute local training with (adjust epochs as needed):\
`.venv/bin/python training.py --data_dir ./data --model_dir ./model_output --epochs 1`

execute sagemaker training by running the sm_estimator.py script 
sagemaker requirements:
- set AWS credentials in environment variables
- set ARN in .env file
- edit S3 bucket in sm_estimator.py to match where you put the data
