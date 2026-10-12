import numpy as np
import pandas as pd
import re

def read_quest(file_path, quest, order = 'original'):
    """Read questionnaires from txt and organize in dictionary
    Args:
        file_path: Path to questionnaire
        quest: specifying whether mfq or macq 

    Returns: 
        dictionary of questionnaires parts
            instructions: Instructions for the first and second part of the questionnaires (list)
            scales: list of the respective scales (list) for first and second part of the questoinnaires
            scales_num: numerical scale (str), applies only to MACQ
            items: list of items (list) for first and second part of the questionnaires
            labels_num: Arabic numerals as choice labels
            labels_alpha: Latin lowercase alphabet as choice labels"""

    order_allowed = ['original', 'inverted']
    if order not in order_allowed:
        raise ValueError(f"order must be one of {order_allowed}, got {order}")
    
    # Read file and save as list of lines
    with open(file_path) as file:
        quest_list = file.readlines()

    # Clean 
    quest_list = [re.sub(r'\n', '', line).strip() for line in quest_list]

    # Create dict
    quest_dict = dict()
    if quest == 'mfq':
        quest_dict['instruction'] = quest_list[0]
        quest_dict['questions'] = [quest_list[1], quest_list[24]]
        quest_dict['scales'] = [quest_list[2:8], quest_list[25:31]]
        quest_dict['labels_str'] = [quest_list[2:8], quest_list[25:31]]                # Added to allow easier adjustment in later steps
        quest_dict['labels_str'][0][0] = quest_dict['labels_str'][0][0].split(' (')[0] # Remove further explanation of first option on scale
        quest_dict['labels_str'][0][5] = quest_dict['labels_str'][0][5].split(' (')[0]
        quest_dict['items'] = [quest_list[8:24], quest_list[31:]]
    elif quest == 'macq':
        quest_dict['instruction'] = quest_list[0]
        quest_dict['questions'] = [quest_list[1], quest_list[29]]
        quest_dict['scales_num'] = ['0-100']*2 
        quest_dict['scales'] = [quest_list[2:8], quest_list[30:35]]
        quest_dict['labels_str'] = [quest_list[2:8], quest_list[30:35]]
        quest_dict['items'] = [quest_list[8:29], quest_list[35:]]
    
    # Arabic numerals 
    quest_dict['labels_num'] = [list(map(str, range(1, len(i)+1))) for i in quest_dict['scales']]

    # Latin alphabet (lowercase)
    quest_dict['labels_alpha'] = [list(map(chr, range(65, 65+len(i)))) for i in quest_dict['scales']] # 97 for lowercase

    if order == 'inverted':
        quest_dict['scales'] = [list(reversed(quest_dict['scales'][part])) for part in range(2)]
        quest_dict['labels_str'] = [list(reversed(quest_dict['labels_str'][part])) for part in range(2)] 

    return quest_dict