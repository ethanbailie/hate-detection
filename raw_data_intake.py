import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

## loads json dataset into a pandas dataframe
with open('data/raw_data.json', 'r') as f:
    data = json.load(f)

records = []
for post_id, post_content in data.items():
    ## turn the individual words into one string for the text
    post_tokens = post_content['post_tokens']
    text = ' '.join(post_tokens)

    annotators = post_content['annotators']
    rationales = post_content['rationales']

    for annotator, rationale in zip(annotators, rationales):
        records.append({
            'post_id': post_id,
            'text': text,
            'label': annotator['label'],
            'annotator_id': annotator['annotator_id'],
            'target': annotator['target'],
            'rationale': rationale
        })

df = pd.DataFrame(records)

encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

## transform the labels
df['label_id'] = encoder.transform(df['label'])

## take the majority vote for what each post is classified as
label_votes = df.groupby('post_id')['label_id'].agg(lambda x: x.value_counts().idxmax()).reset_index()
label_votes.rename(columns={'label_id': 'final_label'}, inplace=True)

## drops the duplicate rows and merges the final labels back to the main dataframe
df_final = df.drop_duplicates(subset=['post_id']).merge(label_votes, on='post_id')
df_final = df_final[['post_id', 'text', 'final_label', 'annotator_id', 'target', 'rationale']]

## exports the dataframe as a pickle file
df_final.to_pickle('data/formatted_data.pkl')
