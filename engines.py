from tqdm.auto import tqdm
import torch
import numpy as np


###############
#   Relation Only Model Training
###############

def postprocess_rel(predictions, labels):
    sig = torch.nn.Sigmoid()
    preds = [[1 if logit>0.5 else 0 for logit in logits] for logits in sig(predictions).detach().cpu().clone().numpy()]
    if labels!='':
        actual = labels.detach().cpu().clone().numpy().astype(int)
    else:
        actual = ''
    return preds, actual


def get_predictions(true_predictions, true_labels, rel_id2label, mode = 'eval'):
    preds = []
    acts = []
    for row in true_predictions:
        row_preds = []
        for i in range(len(row)):
            if row[i] == 1:
                row_preds.append(rel_id2label[i])
        preds.append(' '.join(sorted(row_preds)))

    if mode == 'eval':
        for row in true_labels:
            row_preds = []
            for i in range(len(row)):
                if row[i] == 1:
                    row_preds.append(rel_id2label[i])
            acts.append(' '.join(sorted(row_preds)))
    return preds, acts

def rel_predict_on_batch(dataloader, model, tokenizer, rel_id2label):
    final_preds = []
    final_acts = []
    for batch_idx, batch in enumerate(dataloader):
        _, rel_out, rel_actual = model(batch, task = 'rel')
        predictions, labels = postprocess_rel(rel_out, rel_actual)
        preds, acts = get_predictions(predictions, labels, rel_id2label)
        # if '' in acts:
        #     i = (acts.index(''))
        #     print(tokenizer.decode(batch['input_ids'][i]), batch['rel_labels'][i], rel_actual[i], labels[i])
        final_preds.append(preds)
        final_acts.append(acts)
    
    final_preds = [i for j in final_preds for i in j]
    final_acts = [i for j in final_acts for i in j]
        
    c = []
    for i in zip(final_preds, final_acts):
        if i[0]==i[1]:
            c.append(1)
        else:
            c.append(0)
            #print(i)
    accuracy = np.mean(c)
    
    return final_preds, final_acts, accuracy
    
def rel_trainer(model, tokenizer, rel_id2label, train_dataloader, accelerator, optimizer, lr_scheduler, eval_dataloader, num_training_steps, num_train_epochs, early_stopping_steps, device):

    '''
    REL PREDICTION
    '''

    progress_bar = tqdm(range(num_training_steps))
    min_val_loss = float('inf')
    break_counter = 0
    max_no_improve_counter = early_stopping_steps

    for epoch in range(num_train_epochs):
        train_loss=0
        valid_loss =0
        print(f"[Epoch {epoch} / {num_train_epochs}]")
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            batch.to(device)
            loss, rel_out, rel_actual = model(batch, task = 'rel')
            #loss = outputs.loss
            accelerator.backward(loss, retain_graph = True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            progress_bar.update(1)
            progress_bar.set_postfix(loss = train_loss)



        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                loss, rel_out, rel_actual = model(batch, task = 'rel')


            labels = rel_actual
            predictions = rel_out
            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess_rel(predictions_gathered, labels_gathered)
            valid_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))

        _,_, dev_acc = rel_predict_on_batch(eval_dataloader, model, tokenizer, rel_id2label)
        print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Acc: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss,
                dev_acc
                ))
        
            
        
        
        if valid_loss > min_val_loss: #(valid_accuracy/(1+batch_idx))<max_val_acc: #
             break_counter+=1
        else:
            print('Loss improved, saving model..')
            min_val_loss = valid_loss
            # Save and upload
            accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained('checkpoint/rel_model.pt', save_function=accelerator.save)
            torch.save({
                            'model':model
                        }, 'checkpoints/rel/checkpoint.tar')
            if accelerator.is_main_process:
                tokenizer.save_pretrained('checkpoints/rel/tokenizer.pt')
            
            
        if break_counter>max_no_improve_counter:
            print('Stopping Early..')
            break

    return model, optimizer
        
    
###############
#   NER Only Model Training
###############

from datasets import load_metric

metric = load_metric("seqeval")


def postprocess_ner(predictions, labels, ner_id2label):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[ner_id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [ner_id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return  true_predictions, true_labels

def ner_predict_on_batch(dataloader, model, ner_id2label):
    final_preds = []
    final_acts = []
    final_ner_preds = []
    eval_dataloader_iter = iter(dataloader)
    for batch_idx, batch in enumerate(eval_dataloader_iter):
        _, ner_out, ner_actual = model(batch, task = 'ner')
        
        ner_preds = ner_out.argmax(dim=2)
        final_ner_preds.append(ner_preds)#
        predictions, labels = postprocess_ner(ner_preds, ner_actual, ner_id2label)
        final_preds.append(predictions)
        final_acts.append(labels)
    
    final_preds = [i for j in final_preds for i in j]
    final_acts = [i for j in final_acts for i in j]
    final_ner_preds = [i for j in final_ner_preds for i in j]

    final_preds = [' '.join(i) for i in final_preds]
    final_acts = [' '.join(i) for i in final_acts] 
    final_ner_preds = [' '.join([str(j) for j in i.detach().cpu().clone().numpy()]) for i in final_ner_preds]
    c = []
    for idx, i in enumerate(zip(final_acts, final_preds, final_ner_preds)):
        if i[0]==i[1]:
            
            c.append(1)
        else:
            #print(i)
            c.append(0)
    return final_preds, final_acts, np.mean(c)

def ner_trainer(model, tokenizer, ner_id2label, train_dataloader, accelerator, optimizer, lr_scheduler, eval_dataloader, num_training_steps, num_train_epochs, early_stopping_steps, device):
    '''
    NER PREDICTION
    '''

    progress_bar = tqdm(range(num_training_steps))
    min_val_loss = float('inf')
    break_counter = 0
    max_no_improve_counter = early_stopping_steps


    for epoch in range(num_train_epochs):
        train_loss=0
        valid_loss =0
        print(f"[Epoch {epoch} / {num_train_epochs}]")
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            loss, ner_out, ner_actual = model(batch, task = 'ner')
            #loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            progress_bar.update(1)
            progress_bar.set_postfix(loss = train_loss)
        


        # Evaluation
        model.eval()
        for batch in eval_dataloader:
            with torch.no_grad():
                loss, ner_out, ner_actual = model(batch, task = 'ner')



            ner_preds = ner_out.argmax(dim=-1)
            # Necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(ner_preds, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(ner_actual, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)

            true_predictions, true_labels = postprocess_ner(predictions_gathered, labels_gathered, ner_id2label)
            metric.add_batch(predictions=true_predictions, references=true_labels)
            valid_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - valid_loss))

        _,_, dev_acc = ner_predict_on_batch(eval_dataloader, model, ner_id2label)
        print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tValidation Acc: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss,
                dev_acc
                ))

        results = metric.compute()

        print(
            f"epoch {epoch}:",
            {
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },
        )
        
        if valid_loss > min_val_loss: #(valid_accuracy/(1+batch_idx))<max_val_acc: #
             break_counter+=1
        else:
            print('Loss improved, saving model..')
            min_val_loss = valid_loss
            # Save and upload
            accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained('checkpoint/rel_model.pt', save_function=accelerator.save)
            torch.save({
                            'model':model
                        }, 'checkpoints/ner/checkpoint.tar')
            if accelerator.is_main_process:
                tokenizer.save_pretrained('checkpoints/ner/tokenizer.pt')
            
            
        if break_counter>max_no_improve_counter:
            print('Stopping Early..')
            break
        
    return model, optimizer




def mtl_trainer(model, tokenizer, train_dataloader, accelerator, optimizer, lr_scheduler, eval_dataloader, num_training_steps, num_train_epochs, early_stopping_steps, device):
    

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(30):
        train_loss=0
        valid_loss =0
        print(f"[Epoch {epoch} / {num_train_epochs}]")
        # Training
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            rel_loss, rel_out, rel_actual, ner_loss, ner_out, ner_actual = model(batch, task = 'mtl')
            #loss = outputs.loss
            loss = (rel_loss + ner_loss)/2
            accelerator.backward(loss)
            #accelerator.backward(ner_loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            train_loss+= ((1 / (batch_idx + 1)) * (loss.data.item() - train_loss))
            progress_bar.update(1)
            progress_bar.set_postfix(loss = train_loss)
            
        print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                valid_loss
                ))

    return model, optimizer