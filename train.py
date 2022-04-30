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
from engines import rel_trainer, postprocess_rel, rel_predict_on_batch, ner_trainer, postprocess_ner, ner_predict_on_batch, mtl_trainer
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
    
def main(data_path, train_size, model_checkpoint, batch_size, dropout, lr, early_stopping_steps, num_train_epochs, device, seed, task):
    seed_everything(seed)
    print(f'\nYo! Model will run on GPU: {device}\n')
    dataset, ner_id2label, ner_label2id, rel_id2label, rel_label2id = create_splits_and_vocab(data_path, train_size, seed)
    
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    
    print('\nTokenizing Data...')
    ## batch size =5000 because it makes the batches of different lengths and dataloader the doesnt work
    tokenized_dataset = dataset.map(tokenize_batch, fn_kwargs = {'tokenizer':tokenizer, 
                                                        'ner_label2id':ner_label2id, 
                                                            'rel_label2id' :rel_label2id}, batched = True,remove_columns = dataset['train'].column_names, batch_size = 5000)
    
    print('Tokenizing Data Done.\n')
    
    #collate
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    #data loader

    train_dataloader = DataLoader(
        tokenized_dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["val"], collate_fn=data_collator, batch_size=batch_size
    )
    print('\nDataLoaders Created.\n')
    
    print('\nCreating Model')
    model = Model(model_checkpoint, ner_id2label, ner_label2id, rel_id2label, rel_label2id, dropout = dropout, device = device)
    model.to(device)
    print('\nModel Created\n')

    optimizer = AdamW(model.parameters(), lr=lr)
    
    #accelarator
    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    
    #scheduler
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    print('Starting Training..')
    model.to(device)
    if task == 'rel':
        print('Training Relation Model..')
        model, optimizer = rel_trainer(model, tokenizer, rel_id2label, train_dataloader, accelerator, optimizer, 
                                       lr_scheduler, eval_dataloader, num_training_steps, num_train_epochs, early_stopping_steps, device)

    elif task =='ner':
        print('Training Slot Model..')
        model, optimizer = ner_trainer(model, tokenizer, ner_id2label, train_dataloader, 
                                       accelerator, optimizer, lr_scheduler, eval_dataloader, num_training_steps, num_train_epochs, early_stopping_steps, device)
    
    
    elif task =='mtl':
        print('Training MTL Model..')
        model, optimizer = mtl_trainer(model, tokenizer, train_dataloader, accelerator, optimizer, 
                                       lr_scheduler, eval_dataloader, num_training_steps, num_train_epochs, early_stopping_steps, device)
    return model, tokenizer, eval_dataloader



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default = '')
    args = parser.parse_args()
    
    model, tokenizer, eval_dataloader = main('data/', 1800,
        model_checkpoint = "bert-base-uncased",batch_size = 32, dropout = 0.1, lr = 1e-4, early_stopping_steps = 4, num_train_epochs = 30, device = 'cuda:1',seed = 42, task = args.task)