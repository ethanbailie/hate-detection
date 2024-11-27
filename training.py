import argparse
import os
import torch
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
import time
import datetime
import logging

## logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

def main():
    #### setup ####
    ## sagemaker argument parser
    parser = argparse.ArgumentParser()

    ## hyperparams
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)

    ## sagemaker specific arguments
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--num_labels', type=int, default=3)

    ## parse arguments
    args = parser.parse_args()

    ## set whether we can use cuda or if we need to use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    ## label mappings for the model
    label_list = ['hatespeech', 'offensive', 'normal']
    id2label = {id: label for id, label in enumerate(label_list)}
    label2id = {label: id for id, label in enumerate(label_list)}

    ## setup the tokenizer and model to be used
    logger.info("Setting up tokenizer and model")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=args.num_labels,
        output_attentions=False,
        output_hidden_states=False,
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    #### data loading ####
    logger.info("Loading data...")
    ## load tokenized text
    tokenized_data = torch.load(os.path.join(args.data_dir, 'text.pt'))

    ## instantiate the dataset
    dataset = HateSpeechDataset(
        input_ids=tokenized_data['input_ids'],
        attention_masks=tokenized_data['attention_masks'],
        labels=tokenized_data['labels']
    )

    ## load post_id divisions for train/valid/test split
    with open(os.path.join(args.data_dir, 'post_id_divisions.json'), 'r') as f:
        post_id_divisions = json.load(f)

    ## load processed data
    df_final = pd.read_pickle(os.path.join(args.data_dir, 'formatted_data.pkl'))

    ## create sets for train/valid/test split
    train_ids = set(post_id_divisions['train'])
    valid_ids = set(post_id_divisions['val'])
    test_ids = set(post_id_divisions['test'])

    ## create masks from processed data train/valid/test split
    train_mask = df_final['post_id'].isin(train_ids)
    valid_mask = df_final['post_id'].isin(valid_ids)
    test_mask = df_final['post_id'].isin(test_ids)

    ## get indices for train/valid/test split
    train_indices = df_final[train_mask].index.tolist()
    valid_indices = df_final[valid_mask].index.tolist()
    test_indices = df_final[test_mask].index.tolist()

    ## create subsets
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    test_dataset = Subset(dataset, test_indices)

    ## create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )

    validation_dataloader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size
    )

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size
    )

    logger.info("Data loaded")

    #### training ####
    ## initialize optimizer
    logger.info("Setting up optimizer and scheduler")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)

    ## initialize scheduler
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    ## running the training loop
    for epoch_i in range(args.epochs):
        logger.info(f"\n======== Epoch {epoch_i + 1} / {args.epochs} ========")
        logger.info("Training...")

        ## initialize variables
        t0 = time.time()
        total_train_loss = 0
        model.train()

        ## training loop
        for step, batch in enumerate(train_dataloader):
            ## log progress update every 40 batches
            if step % 40 == 0 and not step == 0:
                elapsed = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
                logger.info(f"  Batch {step} of {len(train_dataloader)}. Elapsed: {elapsed}.")

            ## move batch to device
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            ## zero gradients
            model.zero_grad()

            ## forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels
            )

            ## get loss
            loss = outputs.loss
            total_train_loss += loss.item()

            ## backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            ## update weights
            optimizer.step()
            scheduler.step()

        ## get average training loss
        avg_train_loss = total_train_loss / len(train_dataloader)

        ## get training time
        training_time = str(datetime.timedelta(seconds=int(round(time.time() - t0))))

        ## log results
        logger.info(f"  Average training loss: {avg_train_loss}")
        logger.info(f"  Training epoch took: {training_time}")

        ## evaluate model
        logger.info("Evaluating model...")

        ## initialize eval variables
        t0 = time.time()
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        ## set model to evaluation mode
        model.eval()

        ## evaluate model
        for batch in validation_dataloader:
            ## move batch to device
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            ## no gradient updates
            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels
                )

            ## get loss and logits
            loss = outputs.loss
            logits = outputs.logits

            ## update eval loss
            total_eval_loss += loss.item()

            ## get predictions
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            ## get accuracy
            preds_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            acc = np.sum(preds_flat == labels_flat) / len(labels_flat)
            total_eval_accuracy += acc

        ## get average accuracy
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        logger.info(f"  Accuracy: {avg_val_accuracy}")

        ## get average loss
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        ## get validation runtime
        validation_time = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
        logger.info(f"  Validation Loss: {avg_val_loss}")
        logger.info(f"  Validation took: {validation_time}")
    
    logger.info("\nTraining complete!")

    ## run test set evaluation
    logger.info("\nRunning Test Set Evaluation...")

    ## initialize variables
    all_preds = []
    all_labels = []
    t0 = time.time()

    ## set model to evaluation mode
    model.eval()

    ## test loop
    for batch in test_dataloader:
        ## move batch to device
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        ## no gradient updates
        with torch.no_grad():
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
            )

        ## get logits
        logits = outputs.logits

        ## get predictions
        preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
        labels = b_labels.cpu().numpy()

        ## update all predictions and labels
        all_preds.extend(preds)
        all_labels.extend(labels)

    ## calculate test metrics
    report = classification_report(all_labels, all_preds, digits=4)
    logger.info("\nTest Classification Report:")
    logger.info(report)

    ## get test runtime
    test_time = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
    logger.info(f"  Test evaluation took: {test_time}")

    ## save the model
    logger.info("\nSaving the model...")
    output_dir = os.path.join(args.model_dir, 'model')
    os.makedirs(output_dir, exist_ok=True)

    ## save the model
    model_to_save = model.module if hasattr(model, 'module') else model  # Handle multi-GPU

    ## save the model and tokenizer
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Model saved to {output_dir}")

if __name__ == '__main__':
    main()