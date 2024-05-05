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
    options_letters = ['A', 'B', 'C', 'D']
    ans_id = options_letters.index(doc['answer'])
    
    logprobs = [res[0] for res in results]
    answer_log = logprobs[ans_id]
    answer_perplexity = np.exp(-answer_log)
    pred_indx = np.argmax(logprobs)
    
    results.sort(key=lambda x: -x[0])
    acc = int(pred_indx == ans_id)
    
    target_logprob = logprobs[0]
    if logprobs[0] == answer_log:
        target_logprob = logprobs[1]
        
    margin = np.exp(-target_logprob)-answer_perplexity
    res = {
        'acc': acc,
        'perplexity': answer_perplexity,
    }
    
    if margin >= 0:
        res['pos_margin'] = margin
    else:
        res['neg_margin'] = -margin
    return res
