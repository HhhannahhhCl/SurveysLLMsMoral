import torch
import time
from tqdm import tqdm

from generate import apply_chat_template_ids, item_prompt, quest_scales

"""Batched generation-based methods"""

##### Model parameters #####
def _model_gen_kwargs(model, decoding, eos_id_single=None):
    name = model.name_or_path.split('/')[-1]

    if name in ('normistral-7b-warm', 'normistral-11b-warm'):
        kwargs = dict(max_new_tokens=64)
        kwargs.update(dict(do_sample=False, temperature=0.0) if decoding == 'greedy'
                      else dict(do_sample=True, top_k=64, top_p=0.9, temperature=0.3))
        if eos_id_single is not None:
            kwargs['eos_token_id'] = eos_id_single  # early stop at '\n', per-row, batch-safe
        return kwargs

    if name == 'normistral-7b-warm-instruct':
        kwargs = dict(max_new_tokens=1024, repetition_penalty=1.0, use_cache=True)
        kwargs.update(dict(do_sample=False, temperature=0.0) if decoding == 'greedy'
                      else dict(do_sample=True, top_k=64, top_p=0.9, temperature=0.3))
        return kwargs

    if name == 'normistral-11b-thinking':
        kwargs = dict(max_new_tokens=2048, repetition_penalty=1.0, use_cache=True)
        kwargs.update(dict(do_sample=False, temperature=0.0) if decoding == 'greedy'
                      else dict(do_sample=True, top_k=64, top_p=0.9, temperature=0.3))
        return kwargs

    # default branch: instruct / Qwen / thinking / plain models
    kwargs = dict(max_new_tokens=1024)
    kwargs.update(dict(do_sample=False, temperature=0.0) if decoding == 'greedy'
                  else dict(do_sample=True))
    return kwargs

def _print_conversation(conversation_log):
    """Print the assembled prompt/response log per setting."""
    for key, turns in conversation_log.items():
        label, inst, cloze = key
        print(f"=== setting: label={label!r}, inst={inst}, cloze={cloze} ===")
        for turn in turns:
            print(turn)
        print()

def _trim_pad_tail(row_1d, pad_id):
    """Trim everything from the first pad_id onward"""
    pad_positions = (row_1d == pad_id).nonzero(as_tuple=True)[0]
    if pad_positions.numel() == 0:
        return row_1d
    first_pad = pad_positions[0].item()
    if first_pad == 0:
        # degenerate case: generated nothing before padding kicked in
        return row_1d[:1]
    return row_1d[:first_pad]


def _parse_output_row(model, tokenizer, new_tokens_row, device):
    """Parsing output depending on thinking mode or not"""
    name = model.name_or_path.split('/')[-1]
    out_row = new_tokens_row.unsqueeze(0)

    if 'Qwen3-0.6B' in name:
        out_row = out_row.clone()
        out_row[out_row == 151643] = 151645  # <|endoftext|> -> <|im_end|>

    output_str = tokenizer.decode(out_row[0]).strip()

    if '<think>' in output_str:
        try:
            results = {
                'reasoning': output_str.split("</think>")[0].lstrip("<think>").strip(),
                'response': output_str.split("</think>")[-1],
            }
            think_end = tokenizer('</think>', return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
            idx = (out_row[0] == think_end[0]).nonzero()[0].item()
            next_response = out_row[0, idx + 1:].unsqueeze(0)
        except (IndexError, RuntimeError):
            results = {'reasoning': output_str.lstrip("<think>").strip(), 'response': 'no answer'}
            next_response = torch.tensor([[tokenizer.eos_token_id]], device=out_row.device, dtype=out_row.dtype)
    else:
        results = {'reasoning': 0, 'response': output_str}
        next_response = out_row

    return results, next_response, output_str


##### Prompt Formatting #####
def _build_context(quest, part, question, model, tokenizer, device, prev_context, reasoning=False):
    name = model.name_or_path.split('/')[-1]
    is_chat = any(spec in name for spec in ['instruct', 'thinking', 'Qwen', 'it', 'borealis'])

    if prev_context is None:  # part == 0, or 'single' conv_type (each part starts fresh)
        if is_chat:
            if reasoning:
                context = apply_chat_template_ids(
                    tokenizer,
                    [{'role': 'system', 'content': quest['instruction']},
                     {'role': 'user', 'content': question}],
                    add_generation_prompt=False, device=device, enable_thinking=True,
                )
            else:
                context = apply_chat_template_ids(
                    tokenizer,
                    [{'role': 'system', 'content': quest['instruction']},
                    {'role': 'user', 'content': question}],
                    add_generation_prompt=False, device=device, enable_thinking=False,
                )
            if any(spec in model.name_or_path.split('/')[-1] for spec in ['gemma-3', 'borealis-']):
                return context.squeeze()[:-2].unsqueeze(0)
            return context
        part_header = quest['instruction'] + '\n' + question
        return tokenizer(part_header, return_tensors='pt')['input_ids'].to(device)
 
    # part == 1, multiturn: extend the running context
    linebreak_id = torch.tensor([[214]]).to(device) if model.name_or_path.split('/')[-1] == 'normistral-11b-warm' else tokenizer('\n', return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    if is_chat:
        part2_tok = apply_chat_template_ids(
            tokenizer, [{'role': 'user', 'content': question}],
            add_generation_prompt=False, device=device, enable_thinking=False,
        )
        if any(spec in model.name_or_path.split('/')[-1] for spec in ['gemma-4', 'borealis2']):
            part2_tok = part2_tok.squeeze()[1:].unsqueeze(0)
        elif any(spec in model.name_or_path.split('/')[-1] for spec in ['it', 'borealis']):
            part2_tok = part2_tok.squeeze()[1:-2].unsqueeze(0)
        return torch.cat([prev_context, part2_tok], dim=1)

    part2_tok = tokenizer(question, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    return torch.cat([prev_context, linebreak_id, part2_tok], dim=1)


def _left_pad(rows, pad_id):
    """Left-side padding according to longest sequence"""
    max_len = max(r.size(1) for r in rows)
    padded = []
    for r in rows:
        gap = max_len - r.size(1)
        if gap:
            pad = torch.full((1, gap), pad_id, dtype=r.dtype, device=r.device)
            r = torch.cat([pad, r], dim=1)
        padded.append(r)
    batch = torch.cat(padded, dim=0)
    attention_mask = (batch != pad_id).long()
    mm_token_type_ids = torch.zeros(batch.size(),
                                    dtype=torch.long,
                                    device=batch.device)
    return batch, attention_mask, mm_token_type_ids


@torch.no_grad()
def generate_batch(
    quest, model, tokenizer, device,
    settings,               # list of dicts: {'label': str, 'inst': bool, 'cloze': bool}
    conv_type='multiturn',  
    decoding='greedy',
    max_batch=None,         
    reasoning=False,
    language='nor',
    verbose=False,
):
    if conv_type not in ('single', 'multiturn'):
        raise ValueError(f"conv_type must be 'single' or 'multiturn', got {conv_type!r}")
 
    pad_id = tokenizer.pad_token_id
    linebreak_id = torch.tensor([[214]]).to(device) if model.name_or_path.split('/')[-1] == 'normistral-11b-warm' else tokenizer('\n', return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    is_chat = any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen', 'it', 'borealis'])
    eos_id_single = linebreak_id[0, 0].item() if model.name_or_path.split('/')[-1] in (
        'normistral-7b-warm', 'normistral-11b-warm') else None

    def key_of(s):
        return (s['label'], s['inst'], s['cloze'])

    results_by_setting = {key_of(s): {'part_0': {}, 'part_1': {}} for s in settings}
    context_by_setting = {key_of(s): None for s in settings}
    conversation_log = {key_of(s): [] for s in settings} # for sanity check printing

    for part in range(2):
        # (re)build each setting's context for this part
        for s in settings:
            key = key_of(s)
            scales, add = quest_scales(quest, s['label'], 'generate', language)
            question = quest['questions'][part] + ' ' + scales[part] + add[part]
            if language=='nor':
                if s['inst']:
                    question = 'Spørsmål: ' + question
                if s['cloze']:
                    question = question.replace(
                        'Angi standpunktet ditt ved å bruke denne skalaen:',
                        'Angi standpunktet ditt ved å fullføre setningene. Bruk denne skalaen:',
                    )
            else:
                if s['inst']:
                    question = 'Question: ' + question
                if s['cloze']:
                    question = question.replace(
                        'Please provide your response using this scale:',
                        'Respond by completing the sentences. Use this scale:',
                    )
            prev = None if (conv_type == 'single' or part == 0) else context_by_setting[key]
            prev_len = prev.size(1) if prev is not None else 0
            print(f'prev_context: {prev_len}')
            print(reasoning)
            context_by_setting[key] = _build_context(quest, part, question, model, tokenizer, device, prev, reasoning)
            print(f"[ctx check] part={part} key={key} prev_len={prev_len} new_len={context_by_setting[key].size(1)} conv_type={conv_type}")
            if part == 1:
                print(tokenizer.decode(context_by_setting[key][0], skip_special_tokens=False))

            # For printing
            delta = context_by_setting[key][:, prev_len:]
            conversation_log[key].append(
                f"--- part {part} instructions/question ---\n"
                + tokenizer.decode(delta[0], skip_special_tokens=False)
            )

        n_items = len(quest['items'][part])

        # waves: (setting, item_idx) pairs to generate
        if conv_type == 'single':
            waves = [[(s, i) for i in range(n_items) for s in settings]]
        else:
            # multiturn: item i+1 depends on item i's answer within the same setting,
            # so item_idx must stay sequential; but every setting is independent, so
            # each item is one batched "wave" across all settings.
            waves = [[(s, i) for s in settings] for i in range(n_items)]

        for wave in waves:
            # Chunks turned into batches below
            chunks = [wave] if not max_batch else [wave[i:i + max_batch] for i in range(0, len(wave), max_batch)]

            for chunk in chunks:
                rows = []
                extras = []
                print("Chunk processing starting", flush=True)
                start=time.time()
                for s, item_idx in tqdm(chunk):
                    key = key_of(s)
                    prompt = item_prompt(quest, part, item_idx, s['inst'], s['cloze'], language)

                    if is_chat:
                        if reasoning:
                            print('we made it to the correct branch')
                            tokenized_prompt = apply_chat_template_ids(
                                tokenizer, [{'role': 'user', 'content': prompt}],
                                add_generation_prompt=True, device=device, enable_thinking=True,
                            )
                        else:
                            tokenized_prompt = apply_chat_template_ids(
                                tokenizer, [{'role': 'user', 'content': prompt}],
                                add_generation_prompt=True, device=device, enable_thinking=False,
                            )
                        if any(spec in model.name_or_path.split('/')[-1] for spec in ['it', 'borealis']):
                            if item_idx == 0 or conv_type=='single':
                                tokenized_prompt = tokenized_prompt.squeeze()[3:].unsqueeze(0)
                                extra = torch.cat([linebreak_id, tokenized_prompt], dim=1)
                            else:
                                extra = tokenized_prompt.squeeze()[1:].unsqueeze(0)
                        else: 
                            extra = tokenized_prompt if (item_idx == 0 and part == 0) or conv_type == 'single' or any(spec in model.name_or_path.split('/')[-1] for spec in ['instruct', 'thinking', 'Qwen']) \
                                else torch.cat([linebreak_id, tokenized_prompt], dim=1)
                    else:
                        tokenized_prompt = tokenizer(prompt, return_tensors='pt',
                                                      add_special_tokens=False)['input_ids'].to(device)
                        extra = torch.cat([linebreak_id, tokenized_prompt], dim=1) if (s['inst'] or s['cloze']) \
                            else torch.cat([linebreak_id, tokenized_prompt, linebreak_id], dim=1)

                    rows.append(torch.cat([context_by_setting[key], extra], dim=1))
                    extras.append(extra)

                end=time.time()
                print(f"Chunk processing done in {end-start} seconds", flush=True)

                print(f"Padding starts", flush=True)
                start=time.time()
                batch, attention_mask, mm_token_type_ids = _left_pad(rows, pad_id)
                input_len = batch.size(1)
                end=time.time()
                print(f"Padding took {end-start} seconds", flush=True)

                print(f"[batch check] part={part} | chunk requested={len(chunk)} rows "
                      f"| batch.shape={tuple(batch.shape)} | attn_mask.shape={tuple(attention_mask.shape)}")
                assert batch.shape[0] == len(chunk), (
                    f"Mismatch: expected batch size {len(chunk)} but got {batch.shape[0]} "
                    f"— check max_batch and wave construction."
                    )

                if model.generation_config is not None:
                    model.generation_config.pad_token_id = pad_id

                gen_kwargs = _model_gen_kwargs(model, decoding, eos_id_single=eos_id_single)
                gen_kwargs['input_ids'] = batch
                gen_kwargs['attention_mask'] = attention_mask
                if any(spec in model.name_or_path.split('/')[-1] for spec in ['gemma-4', 'borealis2']):
                    gen_kwargs['mm_token_type_ids'] = mm_token_type_ids

                print("Starting generation", flush=True)
                start = time.time()
                output_tokens = model.generate(**gen_kwargs) 
                end = time.time()
                print(f"Generation time: {end-start} seconds", flush=True)
                new_tokens = output_tokens[:, input_len:]

                if verbose:
                    print(f'part {part}, wave chunk size {len(chunk)}, input_len {input_len}')

                for row_idx, (s, item_idx) in enumerate(chunk):
                    key = key_of(s)
                    clean_row = _trim_pad_tail(new_tokens[row_idx], pad_id)
                    res, next_response, output_str = _parse_output_row(model, tokenizer, clean_row, device)
                    results_by_setting[key][f'part_{part}'][f'item_{item_idx}'] = res

                    # log prompt + answer for this item, for both conv_types
                    prompt_text = tokenizer.decode(extras[row_idx][0], skip_special_tokens=False)
                    conversation_log[key].append(
                        f"----- part {part} item {item_idx} prompt -----\n{prompt_text}"
                    )
                    conversation_log[key].append(
                        f"----- part {part} item {item_idx} response -----\n{output_str}"
                    )

                    if conv_type == 'multiturn':
                        context_by_setting[key] = torch.cat([rows[row_idx], next_response], dim=1)

    _print_conversation(conversation_log)

    return results_by_setting

