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
