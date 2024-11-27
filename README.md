# hate-detection
using finetuned BERT to detect hate speech

execute local training with (adjust epochs as needed):
.venv/bin/python training.py --data_dir ./data --model_dir ./model_output --epochs 1 --batch_size 4 --learning_rate 2e-5 --num_labels 3