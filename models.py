import torch
from transformers import AutoModel
from torch.nn.functional import one_hot


class Model(torch.nn.Module):
    def __init__(self, model_checkpoint, ner_id2label, ner_label2id, rel_id2label, rel_label2id, dropout, device):
        super().__init__()
        self.base = AutoModel.from_pretrained(
                model_checkpoint
                )
        
        # If want to freeze some layers
        # c = 0
        # for name, param in self.base.named_parameters():
        #     c+=1
        #     if c <= 100:
        #         param.requires_grad = False
        self.rel_id2label = rel_id2label
        self.device = device
        self.ner_fc = torch.nn.Linear(self.base.config.hidden_size, len(ner_id2label))
        self.rel_fc = torch.nn.Linear(self.base.config.hidden_size, len(rel_id2label))
        
        self.dropout = torch.nn.Dropout(dropout)
        self.ner_loss_fn = torch.nn.CrossEntropyLoss(ignore_index = -100)
        self.rel_loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'mean')
        
    def one_hotter(self, batch):
        t = torch.tensor([], device = self.device)
        for i in range(len(batch)):
            b = batch[i][torch.where(batch[i]>=0)[0]]
            t = torch.cat((t, one_hot(b, num_classes = len(self.rel_id2label)).sum(dim = 0).unsqueeze(0)))
        return t
    
    def forward(self, batch, task):
        batch.to(self.device)
        if task == 'rel':
            _ , pools = self.base(input_ids=batch['input_ids'],
              attention_mask=batch['attention_mask'],
              token_type_ids=batch['token_type_ids'],
               return_dict=False)
            # print(hidden.shape)
            # print(pools.shape)

            rel_out = self.rel_fc(self.dropout(pools))

            rel_actual = self.one_hotter(batch['rel_labels'])
            loss = self.rel_loss_fn(rel_out, rel_actual)

            return loss, rel_out, rel_actual
        
        elif task == 'ner':
            hidden, _ = self.base(input_ids=batch['input_ids'],
              attention_mask=batch['attention_mask'],
              token_type_ids=batch['token_type_ids'],
               return_dict=False)
            
            ner_out = self.ner_fc(self.dropout(hidden))
            ner_actual = batch['ner_labels']
            #print(ner_out.shape, ner_actual.shape)
            loss = self.ner_loss_fn(ner_out.permute(0,2,1), ner_actual)
            return loss, ner_out, ner_actual
        
        elif task == 'mtl':
            
            hidden, pools = self.base(input_ids=batch['input_ids'],
              attention_mask=batch['attention_mask'],
              token_type_ids=batch['token_type_ids'],
               return_dict=False)
            
            rel_out = self.rel_fc(self.dropout(pools))
            rel_actual = self.one_hotter(batch['rel_labels'])
            
            ner_out = self.ner_fc(self.dropout(hidden))
            ner_actual = batch['ner_labels']
            #print(ner_out.shape, ner_actual.shape)
            
            rel_loss = self.rel_loss_fn(rel_out, rel_actual)
            ner_loss = self.ner_loss_fn(ner_out.permute(0,2,1), ner_actual)
            
            return rel_loss, rel_out, rel_actual, ner_loss, ner_out, ner_actual
        else:
            print('Choose from the following : rel, ner, mtl')
            return
        