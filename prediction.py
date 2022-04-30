from engines import postprocess_rel, get_predictions
import torch

def predict(examples, model, tokenizer, rel_id2label, task = 'rel'):
    
    if task == 'rel':
        preds = []
        for example in examples:
            example = example.split()
            d = tokenizer(example, is_split_into_words=True)
            d['rel_labels'] = [-101 for i in d['input_ids']]
            d['ner_labels'] = [-101 for i in d['input_ids']]

            for c in d:
                d[c] = torch.Tensor(d[c]).long().unsqueeze(0)
            _, pred, _ = model(d, task = task)

            pred = postprocess_rel(pred, '')[0][0]
            pred, _ = get_predictions([pred], '', rel_id2label, mode = 'test')
            
            preds.append((example, ' '.join(pred)))
            
        return preds
    
    #if task == 'ner':
        