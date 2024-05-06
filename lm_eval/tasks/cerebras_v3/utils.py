import numpy as np


def get_options(doc):
    options = [str(option) for option in doc['options']]
    return options

def doc_to_choice(doc):
    options_letters = ['A', 'B', 'C', 'D']
    options = get_options(doc)
    return options_letters[:len(options)]

def doc_to_text(doc):
    options = get_options(doc)
    options = list(zip(['A', 'B', 'C', 'D'], options))
    options_str = ''.join([f'{op[0]}. {op[1]}\n' for op in options])
    prompt = f"{doc['question'].strip()}\n{options_str}Answer:"
    return prompt

def process_results(doc, results):
    # get answer
    options_letters = ['A', 'B', 'C', 'D']
    ans_id = options_letters.index(doc['answer'])
    
    # log probabilities and softmaxes
    logprobs = [res[0] for res in results]
    softmaxes = (np.exp(logprobs)/sum(np.exp(logprobs))).tolist()
    softmaxes += (4-len(softmaxes))*[0]
    
    # perplexity and logprob for answer
    answer_log = logprobs[ans_id]
    answer_perplexity = np.exp(-answer_log)
    pred_indx = np.argmax(logprobs)
    
    # sort to get margins
    results.sort(key=lambda x: -x[0])
    acc = int(pred_indx == ans_id)
    
    target_logprob = logprobs[0]
    if logprobs[0] == answer_log:
        target_logprob = logprobs[1]
        
    margin = np.exp(-target_logprob)-answer_perplexity
    res = {
        'acc': acc,
        'perplexity': answer_perplexity,
        'roc_auc': (softmaxes, ans_id),
    }
    
    if margin >= 0:
        res['pos_margin'] = margin
    else:
        res['neg_margin'] = -margin
    return res

def roc_auc_score(y_true, y_score):
    thresholds = np.unique(y_score)
    thresholds = np.append(thresholds, max(thresholds) + 1)
    
    tpr = []
    fpr = []
    rows = y_score.shape[1]
    
    for thresh in thresholds:
        preds = y_score >= thresh
        
        # True positive: correctly predicted positive
        tp = np.sum(preds[np.arange(len(y_true)), y_true])
        
        # False positive: wrongly predicted as positive
        # True negative: correctly predicted negative
        fp, tn = 0, 0
        for i in range(len(y_true)):
            for row in range(rows):
                if row != y_true[i]:
                    fp += preds[i][row]
                    tn += ~preds[i][row]
        
        # False negative: wrongly predicted as negative
        fn = np.sum(~preds[np.arange(len(y_true)), y_true])
        
        # Compute TPR and FPR
        tpr.append(tp / (tp + fn) if (tp + fn) != 0 else 0)
        fpr.append(fp / (fp + tn) if (fp + tn) != 0 else 0)
        
    # Sort rates by increasing FPR
    sorted_indices = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_indices]
    tpr = np.array(tpr)[sorted_indices]
    
    # Calculate AUC using the trapezoidal rule
    auc = np.trapz(tpr, fpr)
    return auc

def aggregate_auc(results):
    y_true = np.array([res[1] for res in results])
    y_score = np.array([res[0] for res in results])
    roc_auc = roc_auc_score(y_true, y_score)
    return roc_auc
