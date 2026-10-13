import pandas as pd
import numpy as np
import torch
import re
import time
import os
import random
import numpy as np
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed_value=8):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

##### Version handling of transformers #####
def apply_chat_template_ids(tokenizer, messages, add_generation_prompt, device, enable_thinking=None, **kwargs):
    """Wrapper around apply_chat_template to always return a plain input_ids
    tensor regardless of transformers version"""
    call_kwargs = dict(
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_tensors='pt',
        return_dict=False,   # returns tensor
    )
    if enable_thinking is not None:
        call_kwargs['enable_thinking'] = enable_thinking
    call_kwargs.update(kwargs)

    out = tokenizer.apply_chat_template(messages, **call_kwargs)

    if isinstance(out, torch.Tensor):
        return out.to(device)
    return out['input_ids'].to(device)
    
##### SCORING #####
def create_batch(context, candidates, tokenizer, device):  
    """Creating a batch to save forward passes
    Args:
        context: tensor of input_ids
        candidates: list of tensors of input_ids
        tokenizer: respective tokenizer
        
    Returns:
        batch of context + respective response option as sequences of token_ids
        attention_mask: masked out padding tokens
        context_mask: masked out context & padding tokens"""

    if isinstance(context, list):
        context_len = context[0].size(1)
        seqs = [torch.cat([context[0], cand, context[1]], dim=1) for cand in candidates]
    else:
        context_len = context.size(1)
        seqs = [torch.cat([context, cand], dim=1) for cand in candidates]

    batch = pad_sequence(
        [seq.squeeze(0) for seq in seqs],      
        batch_first=True,                       
        padding_value=tokenizer.pad_token_id,
        ).to(device)   

    # Attention mask for logits
    attention_mask = (batch != tokenizer.pad_token_id).long() 

    # Context mask: Masking out everything except for the candidates
    B, S = batch.shape
    cand_len = torch.tensor([cand.size(1) for cand in candidates], device=device)
    pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S) 

    context_mask = (
        (pos >= context_len) &                      
        (pos < context_len + cand_len.unsqueeze(1)) 
    ).long() 

    return batch, attention_mask, context_mask

def score_candidates(candidates, model, tokenizer, device, context=None, verbose=False):
    """
    Return probability, perplexity, and pmi of multi-token candidates given context.

    Args:
        context: Tensor of token ids in context
        candidates: list of tensors of token ids in each candidate
        model
        tokenizer
        device

    Returns: dict of respective values
        mean_probs: mean probabilities of all options over entire vocabulary (not normalized)
        const_prob: constrained probability distribution over candidates summing to 1 (mean_log_probs normalized)
        ppl: perplexity of candidates
    """
    
    if context is None:
        # Add BOS token to compute logprobs of candidates w/o context
        bos_id = tokenizer.bos_token_id
        cand_wo_context = [
            torch.cat([torch.tensor([[bos_id]], device=cand.device), cand], dim=1)
            for cand in candidates
        ]
        batch = pad_sequence(
            [cand.squeeze(0) for cand in cand_wo_context],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
            ).to(device)
        attention_mask = (batch != tokenizer.pad_token_id).long()
        mask = attention_mask[:, 1:]
    else:
        batch, attention_mask, context_mask = create_batch(context, candidates, tokenizer, device)
        mask = context_mask[:, 1:]

    if verbose:
        print(attention_mask)
        # print(context_mask)
        print(mask)

    with torch.no_grad():
        logits = model(input_ids=batch, attention_mask=attention_mask)['logits'] 

    if verbose:
        print(f'logits.size(): {logits.size()}')

    log_probs = F.log_softmax(logits, dim=-1)   

    # Gather next_token_ids + throw out last token pred
    next_token_ids = batch[:, 1:]               
    log_probs = log_probs[:, :-1, :]           

    # Indexing the correct log_probs
    log_probs = log_probs.gather(               
        dim = -1,                               
        index = next_token_ids.unsqueeze(-1)   
    ).squeeze(-1)                              

    sum_log_probs = (log_probs * mask).sum(dim=1) 
    num_tokens = mask.sum(dim=1)
    
    if verbose:
        print(f'sum_log_probs: {sum_log_probs}')
        print(f'num_tokens: {num_tokens}')

    mean_log_probs = sum_log_probs / num_tokens
    mean_probs = torch.exp(mean_log_probs) 
    const_prob = F.softmax(mean_log_probs, dim=0) 
    ppl = torch.exp(-mean_log_probs) 

    return {
        'mean_log_probs': mean_log_probs.tolist(),
        'mean_probs': mean_probs.tolist(),
        'const_prob': const_prob.tolist(),
        'ppl': ppl.tolist()
        }

##### PROMPT #####
def quest_scales(quest, label, scoring, language='nor'):
    """Return converted scales and instruction"""
    if label == 'labels_str':
        scales = [', '.join(quest['scales'][part]) for part in range(2)]
        add = ['.','.']
    elif label in ('labels_num', 'labels_alpha'):
        scales = [', '.join([f"{quest[label][part][i]} - {quest['scales'][part][i]}" for i in range(len(quest['scales'][part]))]) for part in range(2)]
        if language == 'nor':
            fill = 'det tilsvarende tallet' if label == 'labels_num' else 'den tilsvarende bokstaven'
            add = [f'. Oppgi {fill} fra {quest[label][part][0]} til {quest[label][part][-1]}.' for part in range(2)]
        else:
            fill = 'the corresponding number' if label == 'labels_num' else 'the corresponding letter'
            add = [f'. Provide {fill} from {quest[label][part][0]} to {quest[label][part][-1]}.' for part in range(2)]
    if scoring == 'generate':
        if language == 'nor':
            add = [s + ' Oppgi nøyaktig ett av svarene ovenfor. Ikke gi en forklaring.' for s in add]
        else:
            add = [s + ' Provide exactly one of the answers above. Do not provide an explanation.' for s in add]
        print(add)

    return scales, add

def item_prompt(quest, part, item_idx, inst, cloze, language='nor'):
    """Returns item text
    Pure item text OR
    inst=True: including Overveielse: / Utsagn: & Svar: OR
    cloze=True: e.g. 
        Part 1: Om noen gjorde noe for å beskytte familien sin eller ikke er {response}
        Part 2: I at folk bør være villige til å gjøre hva som helst for å hjelpe et familiemedlem er jeg {response}
    """
    text = quest['items'][part][item_idx]

    if language == 'nor':
        if inst:
            return f'Overveielse: {text}\nSvar:' if part == 0 else f'Utsagn: {text}\nSvar:'
        
        if cloze:
            if part == 0:
                return f'{text} er'
            else:
                return f'I at {text.lower()} er jeg'
        return text
    else: 
        if inst:
            return f'Consideration: {text}\nResponse:' if part == 0 else f'Statement: {text}\nResponse:'
        
        if cloze:
            if part == 0:
                return f'{text} is'
            else:
                return f'That {text.lower()} is something I'
        return text
    
##### PROCESS ITEM #####
def item_turn(context, candidates, pmi_norm, quest, part, item_idx, 
    model, tokenizer, device,
    conv_type, inst, cloze, scoring, linebreak_id, language='nor', verbose=False):

    prompt = item_prompt(quest, part, item_idx, inst, cloze, language)
    if any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen', 'it', 'borealis']):
        tokenized_prompt = apply_chat_template_ids(
            tokenizer, [{'role': 'user', 'content': prompt}],
            add_generation_prompt=True, device=device, enable_thinking=False,
        )

        if any(spec in model.name_or_path.split('/')[-1] for spec in ['gemma-4', 'borealis2']):
            extra = tokenized_prompt.squeeze()[1:].unsqueeze(0)

        elif any(spec in model.name_or_path.split('/')[-1] for spec in ['it', 'borealis']):
            if item_idx == 0:
                tokenized_prompt = tokenized_prompt.squeeze()[3:].unsqueeze(0)
                extra = torch.cat([linebreak_id, tokenized_prompt], dim=1)
            else:
                extra = tokenized_prompt.squeeze()[1:].unsqueeze(0)
        else:
            extra = tokenized_prompt if (item_idx == 0 and part == 0) or conv_type == 'single' or any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen']) \
                else torch.cat([linebreak_id, tokenized_prompt], dim=1) 
    else:
        tokenized_prompt = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
        extra = torch.cat([linebreak_id, tokenized_prompt], dim=1) if inst or cloze else torch.cat([linebreak_id, tokenized_prompt, linebreak_id], dim=1)

    score_ctx = torch.cat([context, extra], dim=1) 

    results = score_candidates(candidates, model, tokenizer, device, context=score_ctx, verbose=verbose)
    results['pmi_scores'] = [mlp-pmi_norm[i] for i, mlp in enumerate(results['mean_log_probs'])]

    best_idx = np.argmax(results['pmi_scores']) if scoring == 'pmi' else np.argmin(results['ppl'])

    next_response = candidates[best_idx]

    return results, next_response, score_ctx

####Combinining all prerequisites#####  
def generate(
    quest, model, tokenizer, device, 
    label = 'labels_str', 
    conv_type='multiturn', 
    scoring='pmi', 
    inst=False, 
    cloze=False,
    decoding='greedy',
    language='nor',
    verbose=False): 
    """Generating the responses
    Args:
        quest: Questionnaire in dict format
        model: Loaded model
        tokenizer: Respective loaded tokenizer
        device: Specified device
        label: Potential candidates for the model to respond to items with 
               (Note: The original scale is part of the instructions regardless)
                labels_str - choice option in the form of original scale
                labels_num - Labels for scale in Arabic numerals                            
                labels_alpha - Labels for scale in Latin alphabet
        conv_type:  multiturn (default) - extends context with most likely answer on the scale to mimic conversation
                    single - assumes each item is given individually one by one (different conversations)
        order:  original - Original order of scale from irrelevant/disagree to relevant/agree
                inverted - Inverted order of the scale
    """
    # ValueErrors
    if conv_type not in ('single', 'multiturn'):
        raise ValueError(f"conv_type must be 'single' or 'multiturn', got {conv_type!r}")

    if label not in ('labels_str', 'labels_num', 'labels_alpha'):
        raise ValueError(f"label must be 'labels_str', 'labels_num' or 'labels_alpha', got {label!r}")

    if scoring not in ('pmi', 'ppl', 'generate'):
        raise ValueError(f"scoring must be 'pmi', 'ppl', or 'generate', got {scoring!r}")

    if model.name_or_path.split('/')[-1] == 'normistral-11b-warm':
        linebreak_id = torch.tensor([[214]]).to(device)
    else:
        linebreak_id = tokenizer('\n', return_tensors='pt', add_special_tokens=False)['input_ids'].to(device) 

    if any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'Qwen']):
        eos = tokenizer('<|im_end|>', return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    elif any(spec in model.name_or_path.split('/')[-1] for spec in ['gemma-4', 'borealis2']):
        eos = tokenizer('<turn|>', return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    else:
        eos = tokenizer('<end_of_turn>', return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)


    if language == 'nor':
        fill = 'Spørsmål: ' if inst else ''
    else:
        fill = 'Question: ' if inst else ''
    scales, add = quest_scales(quest, label, scoring, language)    

    results_dict = dict()

    for part in range(2):
        results_dict[f'part_{part}'] = dict()

        question = fill + quest['questions'][part] + ' ' + scales[part] + add[part]
        if cloze:
            # Change instruction to finish sentences
            if language == 'nor':
                question = question.replace(
                    'Angi standpunktet ditt ved å bruke denne skalaen:',
                    'Angi standpunktet ditt ved å fullføre setningene. Bruk denne skalaen:') 
            else:
                question = question.replace(
                    'Please provide your response using this scale:',
                    'Respond by completing the sentences. Use this scale:')

        if conv_type == 'single' or part == 0:
            # For single turn conversation or the very start of the conversation
            if any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen', 'it', 'borealis']):
                # Template for instuction-tuned models (if the tokenizer has this function)
                context = apply_chat_template_ids(
                    tokenizer,
                    [{'role': 'system', 'content': quest['instruction']},
                     {'role': 'user', 'content': question}],
                    add_generation_prompt=False, device=device, enable_thinking=False,
                )

                if any(spec in model.name_or_path.split('/')[-1] for spec in ['gemma-3', 'borealis-']):
                    context = context.squeeze()[:-2].unsqueeze(0)
               
            else:
                # For non-instruction-tuned models
                part_header = quest['instruction'] + '\n' + question
                context = tokenizer(part_header, return_tensors='pt')['input_ids'].to(device)
        else: # multiturn & part == 1
            if any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen','it', 'borealis']):
                part2_tok = apply_chat_template_ids(
                    tokenizer, [{'role': 'user', 'content': question}],
                    add_generation_prompt=False, device=device, enable_thinking=False,
                )
                if any(spec in model.name_or_path.split('/')[-1] for spec in ['gemma-4', 'borealis2']):
                    part2_tok = part2_tok.squeeze()[1:].unsqueeze(0)

                elif any(spec in model.name_or_path.split('/')[-1] for spec in ['it', 'borealis']):
                    part2_tok = part2_tok.squeeze()[1:-2].unsqueeze(0)
                
                context = torch.cat([context, part2_tok], dim=1)
            else:
                part2_tok = tokenizer(question, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
                context = torch.cat([context, linebreak_id, part2_tok], dim=1)
        
        # Tokenize candidates
        if any(score in scoring for score in ['pmi', 'ppl']):
            if (model.name_or_path.split('/')[-1] == 'normistral-7b-warm' or 'pt' in model.name_or_path.split('/')[-1]) and (inst or cloze):
                candidates = [tokenizer(f' {cand}', return_tensors='pt', add_special_tokens=False)['input_ids'].to(device) for cand in quest[label][part]]
            else:
                candidates = [tokenizer(cand, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device) for cand in quest[label][part]]
            if verbose:
                print(candidates)
            
            if inst:
                content = 'Svar:' if language == 'nor' else 'Response:'
                if any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen','it', 'borealis']):
                    prefix = apply_chat_template_ids(
                        tokenizer, [{'role': 'user', 'content': content}],
                        add_generation_prompt=True, device=device, enable_thinking=False,
                    )
                    
                else:
                    prefix = tokenizer(f"\n{content}", return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
                pmi_norm = score_candidates(candidates, model, tokenizer, device, context=prefix, verbose=verbose)['mean_log_probs']

            elif 'instruct' in model.name_or_path.split('/')[-1]:
                prefix = tokenizer.apply_chat_template(
                    [{'role': 'assistant', 'content': ''}],
                    tokenize=True, add_generation_prompt=False, enable_thinking=False,
                    return_dict=False,
                )[:-2]
                
                prefix = torch.tensor(prefix).unsqueeze(0).to(device)
                pmi_norm = score_candidates(candidates, model, tokenizer, device, context=prefix, verbose=verbose)['mean_log_probs']
            elif 'Qwen' in model.name_or_path.split('/')[-1]:
                prefix = apply_chat_template_ids(
                    tokenizer, [{}], add_generation_prompt=True, device=device, enable_thinking=False,
                )
                
                pmi_norm = score_candidates(candidates, model, tokenizer, device, context=prefix, verbose=verbose)['mean_log_probs']
            elif any(spec in model.name_or_path.split('/')[-1] for spec in ['it', 'borealis']):
                prefix = tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': ''}],
                    tokenize=True, add_generation_prompt=True,
                    return_dict=False,
                )
                if any(spec in model.name_or_path.split('/')[-1] for spec in ['gemma-4', 'borealis2']):
                    prefix = prefix[-7:]
                else:
                    prefix = prefix[-3:]
                prefix = torch.tensor(prefix).unsqueeze(0).to(device)
                pmi_norm = score_candidates(candidates, model, tokenizer, device, context=prefix, verbose=verbose)['mean_log_probs']

            else:
                pmi_norm = score_candidates(candidates, model, tokenizer, device, context=None, verbose=verbose)['mean_log_probs'] 
            
            if verbose:
                print(f'prefix = {tokenizer.decode(prefix[0])}')
                print(pmi_norm)

            for item_idx in range(len(quest['items'][part])):
                results, next_response, score_ctx = item_turn(
                    context, candidates, pmi_norm,
                    quest, part, item_idx,
                    model, tokenizer, device, 
                    conv_type, inst, cloze, scoring, 
                    linebreak_id, language, verbose,
                )
                results_dict[f'part_{part}'][f'item_{item_idx}'] = results
                
                if conv_type == 'multiturn':
                    if any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen', 'it', 'borealis']):
                        context = torch.cat([score_ctx, next_response, eos, linebreak_id], dim=1)
                    else:
                        context = torch.cat([score_ctx, next_response], dim=1)
            
                elif conv_type == 'single':
                    if any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen', 'it', 'borealis']):
                        print(tokenizer.decode(torch.cat([score_ctx, next_response, eos], dim=1).squeeze(0)))
                    else:
                        print(tokenizer.decode(torch.cat([score_ctx, next_response], dim=1).squeeze(0)))


    if conv_type == 'multiturn':
        print(tokenizer.decode(context.squeeze(0)))

    return results_dict

def add_info(results_dict, quest, language, model, label, conv_type, scoring, order, inst, cloze, decoding):
    results_dict['quest'] = quest
    results_dict['language'] = language
    results_dict['model'] = model
    results_dict['label'] = label
    results_dict['conv_type'] = conv_type
    results_dict['scoring'] = scoring
    results_dict['order'] = order
    results_dict['inst'] = inst
    results_dict['cloze'] = cloze
    if scoring == 'generate':
        results_dict['decoding'] = decoding
    return results_dict


