import typing

def tokenize_prefix_and_target(tokenizer , prefixes , targets , max_length = None) :
    if tokenizer.pad_token_id is None :
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if prefixes :
        prefix_lengths = [len(s) for s in tokenizer(prefixes , add_special_tokens = False).input_ids]
        tokenized = tokenizer([p + t for p , t in zip(prefixes , targets)] , return_tensors = 'pt' , padding = True , truncation = True , max_length = max_length , add_special_tokens = False)
        labels = tokenized.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        if tokenizer.padding_side == 'left' :
            starts = tokenized.attention_mask.size(-1) - tokenized.attention_mask.sum(-1)
        else :
            starts = [0] * len(prefix_lengths)
        for i in range(len(prefix_lengths)) :
            start = starts[i]
            labels[i , start :start + prefix_lengths[i]] = -100
            labels[i , start] = -100
        tokenized['labels'] = labels
    else :
        tokenized = tokenizer(targets ,
                              return_tensors = 'pt' ,
                              padding = True , truncation = True ,
                              max_length = max_length ,
                              add_special_tokens = False
                              )
        if tokenizer.padding_side == 'left' :
            starts = tokenized.attention_mask.size(-1) - tokenized.attention_mask.sum(-1)
        else :
            starts = [0] * len(targets)
        labels = tokenized.input_ids.detach().clone()
        labels[range(len(targets)) , starts] = -100
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized['labels'] = labels
    return tokenized


def flatten_dict(d) :
    to_process = list(d.items())
    output = {}
    while len(to_process) :
        k , v = to_process.pop()
        if isinstance(v , typing.MutableMapping) :
            to_process.extend([(f"{k}.{k_}" , v_) for (k_ , v_) in v.items()])
        else :
            assert k not in output.keys() , "Somehow ended up with duplicate keys"
            output[k] = v

    return output

def forgetting_questions_split(data,step_size,forgetting_percentage):
    for index,d in enumerate(data):
        if index == 0:
            pass
        else:
            old_forgetting_questions = data[index-1]['forgetting_questions'].copy()
            Rank1_forgetting_questions = []
            Rank2_forgetting_questions = []
            Rank3_forgetting_questions = []
            Rank4_forgetting_questions = []
            for question in data[index-1]['questions']:
                if question['Rank'] == '1':
                    Rank1_forgetting_questions.append(question)
                elif question['Rank'] == '2':
                    Rank2_forgetting_questions.append(question)
                elif question['Rank'] == '3':
                    Rank3_forgetting_questions.append(question)
                elif question['Rank'] == '4':
                    Rank4_forgetting_questions.append(question)
            append_forgetting_questions = []
            append_forgetting_questions.extend(Rank1_forgetting_questions[:int(len(Rank1_forgetting_questions) * forgetting_percentage)])
            append_forgetting_questions.extend(Rank2_forgetting_questions[:int(len(Rank2_forgetting_questions) * forgetting_percentage)])
            append_forgetting_questions.extend(Rank3_forgetting_questions[:int(len(Rank3_forgetting_questions) * forgetting_percentage)])
            append_forgetting_questions.extend(Rank4_forgetting_questions[:int(len(Rank4_forgetting_questions) * forgetting_percentage)])
            old_forgetting_questions.extend(append_forgetting_questions)
            new_forgetting_questions = old_forgetting_questions.copy()
            d['forgetting_questions'] = new_forgetting_questions

    return data







