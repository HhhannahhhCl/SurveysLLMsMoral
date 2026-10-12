import warnings
import json
import time
import gc
import os
import argparse
import random
import torch

from read_data import read_quest
from generate import set_seed, generate, add_info
from generate_batched import generate_batch

warnings.filterwarnings('ignore')

### ARGS ###
parser = argparse.ArgumentParser()

# General
parser.add_argument("--output_dir", type=str, help="Output directory.")  
parser.add_argument("--model", default="ltg/normistral-7b-warm-instruct", type=str, help="Model")
parser.add_argument("--hf", type=str, help="HF_HOME") 
parser.add_argument("--language", default="nor",type=str,help="Language of questionnaire")
parser.add_argument("--generate", action="store_true", help="Generation included (default: False meaning only scoring)")
parser.add_argument("--decoding", default="greedy", type=str, help="Greedy decoding (greedy) or default temperature (sampling)")
parser.add_argument("--batch_size", default=8, type=int,help="Batch size")
parser.add_argument("--reasoning", action="store_true", help="Reasoning (thinking mode) enables (default: False)")
parser.add_argument("--verbose", action="store_true", help="Sanity checks enabled (default: False)")

 
args = parser.parse_args()

os.environ['HF_HOME'] = args.hf
os.environ['HF_HUB_CACHE'] = f"{args.hf}/hub"

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoTokenizer,
    AutoProcessor
)

local_rank = args.gpu_rank if args.gpu_rank is not None else int(os.environ.get("SLURM_LOCALID", 0))
num_shards = args.num_shards
torch.cuda.set_device(0)

print(f"Starting script with {args.model}, decoding = {args.decoding}")

# Sampling random seed
set_seed()
seeds = random.sample(range(2**32), 10) 

quest = ['mfq', 'macq'] 
label = ['labels_alpha', 'labels_num', 'labels_str']
conv_type = ['single','multiturn'] 
scoring = ['pmi', 'ppl']
order = ['original', 'inverted']

out_path = args.output_dir

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


tokenizer = AutoTokenizer.from_pretrained(args.model)


model = AutoModelForCausalLM.from_pretrained(
    args.model, 
    device_map= "auto",
    dtype=torch.bfloat16,
    )
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

wd_path = os.path.dirname(os.getcwd())

if any(spec in args.model for spec in ['Qwen']):
    if args.reasoning:
        reas = "with/"
    else:
        reas = "without/"
else:
    reas = ""

def build_settings():
    """Every (label, inst, cloze) setting combo."""
    settings = []
    for l in label:
        settings.append({'label': l, 'inst': False, 'cloze': False})
        settings.append({'label': l, 'inst': True, 'cloze': False})
        if l == 'labels_str':
            settings.append({'label': l, 'inst': False, 'cloze': True})
    return settings

decoding = args.decoding

if args.generate:
    if decoding == 'sampling':
        all_jobs = [(seed, q, o, conv) for seed in seeds for q in quest for o in order for conv in conv_type]
        my_jobs = all_jobs[local_rank::num_shards]
 
        if args.verbose:
            print(f"[shard {local_rank}/{num_shards}] handling {len(my_jobs)} / {len(all_jobs)} jobs")
        print(args.reasoning) 
        for seed, q, o, conv in my_jobs:
            set_seed(seed)
            data = read_quest(f'{wd_path}/SurveysLLMsMoral/questionnaires/{q}_{args.language}.txt', quest=q, order=o)
            settings = build_settings()
 
            results_by_setting = generate_batch(
                data, model, tokenizer, device,
                settings=settings,
                conv_type=conv,
                decoding='sampling',
                max_batch=args.batch_size,
                reasoning=args.reasoning,
                language=args.language,
                verbose=args.verbose,
            )
 
            for (l, inst, cloze), results in results_by_setting.items():
                results = add_info(results, q, args.language, args.model, l, conv,
                                    'generate', o, inst, cloze, decoding='sampling')
                results['seed'] = seed
 
                suffix = ('_inst' if inst else '') + ('_cloze' if cloze else '')
                fname = f'{out_path}/sampling/{args.model}/{reas}{args.model}_{q}_{l}_{o}_{conv}_generate{suffix}_{seed}.json'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                with open(fname, 'w') as f:
                    f.write(json.dumps(results, indent=4, ensure_ascii=False))
                if args.verbose:
                    print(f"[shard {local_rank}] wrote {fname}")

    elif decoding == 'greedy': 
        set_seed()
        all_jobs = [(q, o, conv) for q in quest for o in order for conv in conv_type]
        my_jobs = all_jobs[local_rank::num_shards]
 
        if args.verbose:
            print(f"[shard {local_rank}/{num_shards}] handling {len(my_jobs)} / {len(all_jobs)} jobs")
 
        for q, o, conv in my_jobs:
            data = read_quest(f'{wd_path}/SurveysLLMsMoral/questionnaires/{q}_{args.language}.txt', quest=q, order=o)
            settings = build_settings()
 
            results_by_setting = generate_batch(
                data, model, tokenizer, device,
                settings=settings,
                conv_type=conv,
                decoding='greedy',
                max_batch=args.batch_size,
                reasoning=args.reasoning,
                language=args.language,
                verbose=args.verbose,
            )
 
            for (l, inst, cloze), results in results_by_setting.items():
                results = add_info(results, q, args.language, args.model, l, conv,
                                    'generate', o, inst, cloze, decoding='greedy')
 
                suffix = ('_inst' if inst else '') + ('_cloze' if cloze else '')
                fname = f'{out_path}/greedy/{args.model}/{reas}{args.model}_{q}_{l}_{o}_{conv}_generate{suffix}.json'
                os.makedirs(os.path.dirname(fname), exist_ok=True)
                with open(fname, 'w') as f:
                    f.write(json.dumps(results, indent=4, ensure_ascii=False))
                if args.verbose:
                    print(f"[shard {local_rank}] wrote {fname}")

# Scoring-based
else:
    set_seed()
    decoding = None
    for q in quest: 
        for o in order:
            # Loading data
            data = read_quest(f'{wd_path}/SurveysLLMsMoral/questionnaires/{q}_{args.language}.txt', quest=q, order=o)
            for l in label:
                for conv in conv_type:
                    if conv == 'multiturn':
                        for s in scoring:
                            # Standard
                            results = generate(
                                data, model, tokenizer, device, 
                                label=l, conv_type=conv, scoring=s, inst=False, cloze=False, decoding=decoding,
                                language=args.language, verbose=args.verbose)
                            results = add_info(results, q, args.language, args.model, l, conv, s, o, False, False, decoding=decoding)

                            with open(f'{out_path}/scoring/{args.model}/{args.model}_{q}_{l}_{o}_{conv}_{s}.json', 'w') as f:
                                f.write(json.dumps(results, indent=4, ensure_ascii=False))
                            print(json.dumps(results, indent=4, ensure_ascii=False))

                            # Minimal instruction words
                            results = generate(
                                data, model, tokenizer, device, 
                                label=l, conv_type=conv, scoring=s, inst=True, cloze=False, decoding=decoding,
                                language=args.language, verbose=args.verbose)
                            results = add_info(results, q, args.language, args.model, l, conv, s, o, True, False, decoding=decoding)
                        

                            with open(f'{out_path}/scoring/{args.model}/{args.model}_{q}_{l}_{o}_{conv}_{s}_inst.json', 'w') as f:
                                f.write(json.dumps(results, indent=4, ensure_ascii=False))
                            print(json.dumps(results, indent=4, ensure_ascii=False))

                            if l == 'labels_str':
                                # Cloze-style probing
                                results = generate(
                                    data, model, tokenizer, device, 
                                    label=l, conv_type=conv, scoring=s, inst=False, cloze=True, decoding=decoding,
                                    language=args.language, verbose=args.verbose)
                                results = add_info(results, q, args.language, args.model, l, conv, s, o, False, True, decoding=decoding)
                                

                                with open(f'{out_path}/scoring/{args.model}/{args.model}_{q}_{l}_{o}_{conv}_{s}_cloze.json', 'w') as f:
                                    f.write(json.dumps(results, indent=4, ensure_ascii=False))
                                print(json.dumps(results, indent=4, ensure_ascii=False))

                    else: 
                        results = generate(
                            data, model, tokenizer, device, 
                            label=l, conv_type=conv, scoring=scoring[0], inst=False, cloze=False, decoding=decoding,
                            language=args.language, verbose=args.verbose)
                        results = add_info(results, q, args.language, args.model, l, conv, scoring[0], o, False, False, decoding=decoding)
                        

                        with open(f'{out_path}/scoring/{args.model}/{args.model}_{q}_{l}_{o}_{conv}_{scoring[0]}.json', 'w') as f:
                            f.write(json.dumps(results, indent=4, ensure_ascii=False))
                        print(json.dumps(results, indent=4, ensure_ascii=False))

                        # Minimal instruction words
                        results = generate(
                            data, model, tokenizer, device, 
                            label=l, conv_type=conv, scoring=scoring[0], inst=True, cloze=False, decoding=decoding,
                            language=args.language, verbose=args.verbose)
                        results = add_info(results, q, args.language, args.model, l, conv, scoring[0], o, True, False, decoding=decoding)
                       
                        with open(f'{out_path}/scoring/{args.model}/{args.model}_{q}_{l}_{o}_{conv}_{scoring[0]}_inst.json', 'w') as f:
                            f.write(json.dumps(results, indent=4, ensure_ascii=False))
                        print(json.dumps(results, indent=4, ensure_ascii=False))


                        if l == 'labels_str':
                            # Cloze-style probing
                            results = generate(
                                data, model, tokenizer, device, 
                                label=l, conv_type=conv, scoring=scoring[0], inst=False, cloze=True, decoding=decoding,
                                language=args.language, verbose=args.verbose)
                            results = add_info(results, q, args.language, args.model, l, conv, scoring[0], o, False, True, decoding=decoding)

                            with open(f'{out_path}/scoring/{args.model}/{args.model}_{q}_{l}_{o}_{conv}_{scoring[0]}_cloze.json', 'w') as f:
                                f.write(json.dumps(results, indent=4, ensure_ascii=False))
                            print(json.dumps(results, indent=4, ensure_ascii=False))

print('SCRIPT EXECUTED SUCCESSFUL')
print(f'Model: {args.model}')   
