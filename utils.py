import numpy as np
from datasets import load_metric
#step 7 - Align labels with tokens
def align_labels_with_tokens(labels, word_ids):
    
    new_labels = []
    prev_word = None
    for word_id in word_ids:
        if word_id is None or word_id==prev_word:
            new_labels.append(-100)  #need to experiment with this
            
        else:
            new_labels.append(labels[word_id])
            prev_word = word_id
    return new_labels
            
    
def tokenize_batch(examples, **kwargs):
        
    
    #the padding depends upon the longest length of sentence sent in map
    tokenized_examples = kwargs['tokenizer'](examples['utt'], is_split_into_words=True, truncation=True, padding=True, max_length = 30)
    all_ner_labels = examples['iob']
    ner_labels = []
    rel_labels = []
    for row_id, example in enumerate(all_ner_labels):
        #get word_ids for row_id
        word_ids = tokenized_examples.word_ids(row_id)
        
        #get label2id
        ner_label = [kwargs['ner_label2id'][i] for i in example]
        
        ner_label = align_labels_with_tokens(ner_label, word_ids)
        
        ner_labels.append(ner_label)
        
    for example in examples['rel']:
        rel_label = [kwargs['rel_label2id'][i] for i in example]
        #make labels equal to length of sentence to make compatible with token collate fn
        rel_labels.append(rel_label + (len(ner_label) - len(rel_label))*[-100])
        
    
    tokenized_examples['ner_labels'] = ner_labels
    tokenized_examples['rel_labels'] = rel_labels
    
    
    
    return tokenized_examples
        
    

class MetricCalc:
    def __init__(self, ):
        self.ner_metric = load_metric("seqeval")
        
    #ner metrics
    def compute_ner_metrics(eval_preds):
        """
        This compute_metrics() function first takes the argmax of the logits to convert them to predictions 
        (as usual, the logits and the probabilities are in the same order, so we don’t need to apply the softmax).
        Then we have to convert both labels and predictions from integers to strings. 
        We remove all the values where the label is -100, then pass the results to the metric.compute() method:
        """
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.ner_metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

    

    
    
   