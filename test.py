import os

from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import torch
import numpy as np
import random
from transformers import AdamW, AutoTokenizer,  AutoModel
from torch.nn.functional import one_hot
from collections import Counter
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from data_layer import create_splits_and_vocab 
from utils import tokenize_batch
from models import Model
from torch.optim import AdamW, Adam
from transformers import get_scheduler
from accelerate import Accelerator
import joblib
from engines import rel_trainer, postprocess_rel, rel_predict_on_batch, ner_trainer, postprocess_ner, ner_predict_on_batch
from prediction import predict

from utils import align_labels_with_tokens
def seed_everything(seed: int):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    


def final_prediction(task):
    if task == 'rel':
        data['rel_pred'] = ''
        tokenizer = AutoTokenizer.from_pretrained("checkpoints/rel/tokenizer.pt/")
        checkpoint = torch.load('checkpoints/rel/checkpoint.tar') 
        model = checkpoint['model']
        #model.to('cpu')
        rel_id2label = joblib.load('util_files/rel_id2label.joblib')
        preds = predict(data.utterances.values, model, tokenizer, rel_id2label, task = task)
        rel_preds = []

        for idx, pred in enumerate(preds):
            data.loc[idx, 'rel_pred'] = pred[1]
            #rel_preds.append((pred[0], pred[1], ' '.join(data.loc[idx, 'Core Relations'])))
        
        
        
            
        
        return data
            
    if task == 'ner':
        tokenizer = AutoTokenizer.from_pretrained("checkpoints/ner/tokenizer.pt/")
        checkpoint = torch.load('checkpoints/ner/checkpoint.tar') 
        model = checkpoint['model']
        
        ner_id2label = joblib.load('util_files/ner_id2label.joblib')
        
        preds = []
        data['ner_pred'] = ''
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        for idx, example in enumerate(data.utterances.values):
            example = example.split(' ')

            d = tokenizer(example, is_split_into_words=True)
            #print(d)
            d['rel_labels'] = [-101 for i in d['input_ids']]
            d['ner_labels'] = [0 for i in d['input_ids']]
            #print(tokenizer.convert_ids_to_tokens(d['input_ids']))

            for c in d:
                d[c] = torch.Tensor(d[c]).long().unsqueeze(0)
            _, pred, _ = model(d, task = 'ner')
            pred  = pred.argmax(dim=2)[0].detach().cpu().clone().numpy()

            word_ids = d.word_ids()


            prev_word_id = None
            for ind in range(len(word_ids)):
                if word_ids[ind] is None or prev_word_id == word_ids[ind]:
                    pred[ind] = -100
                else:
                    prev_word_id = word_ids[ind]




            pred = [i for i in pred if i!=-100]

            pred = [ner_id2label[i] for i in pred]
            #print(pred)
            data.loc[idx, 'ner_pred'] = ' '.join(pred)
            
    return data


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default = 'data/hw1_test.xlsx')
    args = parser.parse_args()

    seed_everything(42)
    data = pd.read_excel(args.file_path, engine = 'openpyxl')

    data = final_prediction(task = 'rel')
    data = final_prediction(task = 'ner')
    
    with open('data/predictions.txt', 'w') as f:
        for i in range(len(data)):

            l = data.loc[i][['utterances', 'ner_pred', 'rel_pred']].values
            f.write(l[0] + "\t" + l[1] + "\t" +  l[2] + "\n")
