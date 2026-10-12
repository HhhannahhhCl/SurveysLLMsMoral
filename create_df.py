import json
import os
import numpy as np
import pandas as pd

"""Translating json files to pd.DataFrame"""

quest_prefix = {'mfq': 'MFQ',
                'macq': 'MACQ'}

def get_best(scores, scoring):
    if scoring == 'pmi':
        return int(np.argmax(scores))
    else: # ppl
        return int(np.argmin(scores))

def get_response(quest_dict, scoring):
    prefix = quest_prefix[quest_dict['quest']]
    responses = dict()

    if scoring == 'generate':
        for part in range(2):
            for item_idx in range(len(quest_dict[f'part_{part}'])):
                col = f'{prefix}{part+1}_{item_idx+1}'

                responses[f'{col}_reasoning'] = quest_dict[f'part_{part}'][f'item_{item_idx}']['reasoning']
                responses[f'{col}_response'] = quest_dict[f'part_{part}'][f'item_{item_idx}']['response']

    else:
        for part in range(2):
            for item_idx in range(len(quest_dict[f'part_{part}'])):
                col = f'{prefix}{part+1}_{item_idx+1}'

                response_scores = quest_dict[f'part_{part}'][f'item_{item_idx}']['pmi_scores'] if scoring == 'pmi' else quest_dict[f'part_{part}'][f'item_{item_idx}']['ppl']
                responses[col] = get_best(response_scores, scoring)
                responses[f'{col}_dist'] = response_scores
                responses[f'{col}_prob'] = quest_dict[f'part_{part}'][f'item_{item_idx}']['const_prob']
    
    return responses

def json_rows(path):
    with open(path) as f:
        data = json.load(f)

    base_meta = {
        'quest': data['quest'],
        'language': data['language'],
        'model': data['model'],
        'labels': data['label'],
        'conv_type': data['conv_type'],
        'order': data['order'],
        'inst': data['inst'],
        'cloze': data['cloze'],
    }
    if 'decoding' in data.keys():
        base_meta['decoding'] = data['decoding']
        if data['decoding'] == 'default':
            base_meta['seed'] = data['seed']
        else:
            base_meta['seed'] = 8
    else: 
        base_meta['decoding'] = 'scoring'
        base_meta['seed'] = 8

    if data['conv_type'] == 'single' and any(scoring in data['scoring'] for scoring in ['pmi', 'ppl']):
        rows = []  
        for scoring in ('pmi', 'ppl'):
            meta = {**base_meta, 'scoring': scoring}
            responses = get_response(data, scoring)
            rows.append({**meta, **responses})
        return rows
    else:
        scoring = data['scoring']
        meta = {**base_meta, 'scoring': scoring}
        responses = get_response(data, scoring)
        return [{**meta, **responses}]

def create_df(folder):
    rows = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith('.json'):
            continue
        try:
            rows.extend(json_rows(os.path.join(folder, filename)))
        except Exception as e:
            print(f'Warning: could not parse {filename}: {e}')

    df = pd.DataFrame(rows)

    meta_cols = ['quest', 'language', 'model', 'labels', 'conv_type', 'scoring', 'order', 'inst', 'cloze', 'decoding', 'seed']
    response_cols = sorted(
        [c for c in df.columns if c not in meta_cols],
        key=lambda c: (c.split('_')[0], int(c.split('_')[1]))
    )

    df = df[meta_cols + response_cols]

    output_path = os.path.join(folder, "questionnaire_responses.csv")
    df.to_csv(output_path, index=False)
    return df

        
