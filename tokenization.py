from transformers import BertTokenizer
import pandas as pd
import torch

## initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

## tokenization param
default_max_len = 128

## tokenizer
def tokenize(texts: list[str], max_len: int=default_max_len, tokenizer: BertTokenizer=tokenizer):
    '''
    takes in a list of texts and tokenizes them with the input tokenizer
    this defaults to the bert-base-uncased tokenizer from the transformers library

    Args:
        texts (list of str): texts to tokenize
        max_len (int): max length of the tokenized texts
        tokenizer (BertTokenizer): bert tokenizer
    
    Returns:
        dict: dictionary containing the input ids and attention masks
    '''
    ## initialize lists for input ids and attention masks
    input_ids = []
    attention_masks = []

    ## for every text in the dataset, use the tokenizer to encode it
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        ## append the encoded text and attention mask to their respective lists
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    ## concatenate the lists of input ids and attention masks
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return {'input_ids': input_ids, 'attention_masks': attention_masks}

## load the dataset
df = pd.read_pickle('data/transformed_dataset.pkl')

## tokenize the texts
tokenized_texts = tokenize(df['text'].tolist())

## add encoded labels
tokenized_texts['labels'] = torch.tensor(df['final_label'].values)

## TODO: export this somewhere, then split the tokenized texts into train/test (remember to balance the dataset)
