import os
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import torch
import numpy as np
import random
import joblib

def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    


def create_splits_and_vocab(data_path, train_size, seed):
    seed_everything(seed)
    print(f'Creating Splits with training size {train_size}...')

    data = pd.read_excel(data_path + 'hw1_train.xlsx', engine = 'openpyxl').sample(frac = 1)
    if len(data) <= train_size:
        print(f'Give train_size < total data size that is of length {len(data)}.\nExited!')
        return
    test = pd.read_excel(data_path + 'hw1_test.xlsx', engine = 'openpyxl')
    
    data['utterances'] = data['utterances'].apply(lambda x: [str(i).lower() for i in x.split()])
    data['IOB Slot tags'] = data['IOB Slot tags'].apply(lambda x: x.split())
    data['Core Relations'] = data['Core Relations'].fillna('no_role') #adding no role for blanks
    data['Core Relations'] = data['Core Relations'].apply(lambda x: x.split())
    
    
    #split
    train, val = data.iloc[:train_size], data[train_size:]
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    joblib.dump(val, 'data/val.joblib')
    
    #create dataset
    dataset = DatasetDict({
    'train':Dataset.from_pandas(train),
    'val': Dataset.from_pandas(val),
    })

    dataset = dataset.rename_columns({'utterances':'utt', 'IOB Slot tags':'iob', 'Core Relations': 'rel'})
    
    print('Creating Vocab...')
    #get ner vocab
    ner_id2label = {i:j for i,j in enumerate(set([i for j in dataset['train']['iob'] for i in j]))}
    ner_label2id = {j:i for i, j in ner_id2label.items()}
    print(f"ner_id2label: \n{ner_id2label}\n")
    print(f"ner_label2id: \n{ner_label2id}\n")
    
    #get rel vocab
    rel_id2label = {i:j for i,j in enumerate(set([i for j in dataset['train']['rel'] for i in j]))}
    rel_label2id = {j:i for i, j in rel_id2label.items()}

    print(f"rel_id2label: \n{rel_id2label}\n")
    print(f"rel_label2id: \n{rel_label2id}\n")
    print('Vocab Created..')
    
    #save both vocabs
    print('Saving vocabs to disk...')
    joblib.dump(ner_id2label, 'util_files/ner_id2label.joblib')
    joblib.dump(ner_label2id, 'util_files/ner_label2id.joblib')
    joblib.dump(rel_id2label, 'util_files/rel_id2label.joblib')
    joblib.dump(rel_label2id, 'util_files/rel_label2id.joblib')
    print('Vocab Saved.')
    return dataset, ner_id2label, ner_label2id, rel_id2label, rel_label2id