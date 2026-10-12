import pandas as pd
import numpy as np
import re
from scipy.stats import entropy, f, iqr
from scipy.spatial.distance import jensenshannon, mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_curve, roc_auc_score

import statsmodels
import statsmodels.formula.api as smf
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import qqplot

from typing import Type

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


##### Prerequesites #####
# Answer scales
scales = {
	'MFQ1': [
		'ikke relevant i det hele tatt',
		'ikke særlig relevant',
		'litt relevant',
		'ganske relevant',
		'svært relevant',
		'ekstremt relevant'
	],
	'MFQ2': [
		'helt uenig',
		'ganske uenig',
		'litt uenig',
		'litt enig',
		'ganske enig',
		'helt enig'
	],
	'MACQ1': [
		'ikke relevant i det hele tatt',
		'ikke særlig relevant',
		'litt relevant',
		'ganske relevant',
		'veldig relevant',
		'ekstremt relevant'
	],
	'MACQ2': [
		'svært uenig',
		'uenig',
		'verken enig eller uenig',
		'enig',
		'svært enig'
	]
}

scales_en = {
	'MFQ1': [
		'not at all relevant',
		'not very relevant',
		'slightly relevant',
		'somewhat relevant',
		'very relevant',
		'extremely relevant'
	],
	'MFQ2': [
		'strongly disagree',
		'moderately disagree',
		'slightly disagree',
		'slightly agree',
		'moderately agree',
		'strongly agree'
	],
	'MACQ1': [
		'not at all relevant',
		'not very relevant',
		'slightly relevant',
		'somewhat relevant',
		'very relevant',
		'extremely relevant'
	],
	'MACQ2': [
		'strongly disagree',
		'disagree',
		'neither agree or disagree',
		'agree',
		'strongly agree'
	]
}

# For parsing
response_map_nor = {
	# Numeric strings
	"1": 1,
	"2": 2,
	"3": 3,
	"4": 4,
	"5": 5,
	"6": 6,

	# Integers 
	1: 1,
	2: 2,
	3: 3,
	4: 4,
	5: 5,
	6: 6,

	# Letters
	"A": 1,
	"B": 2,
	"C": 3,
	"D": 4,
	"E": 5,
	"F": 6,

	# Relevance scale
	'ikke relevant i det hele tatt': 1,
	'ikke særlig relevant': 2,
	'litt relevant': 3,
	'ganske relevant': 4,
	'svært relevant': 5,	 # MFQ
	'veldig relevant': 5,	# MACQ 
	'ekstremt relevant': 6,

	# Agreement scale - MFQ 
	'helt uenig': 1,
	'ganske uenig': 2,
	'litt uenig': 3,
	'litt enig': 4,
	'ganske enig': 5,
	'helt enig': 6,

	# Agreement scale - MACQ:
	'svært uenig': 1,
	'uenig': 2,
	'verken enig eller uenig': 3,
	'enig': 4,
	'svært enig': 5,
}

response_map_en_mfq = {
	# Numeric strings
	"1": 1,
	"2": 2,
	"3": 3,
	"4": 4,
	"5": 5,
	"6": 6,

	# Integers 
	1: 1,
	2: 2,
	3: 3,
	4: 4,
	5: 5,
	6: 6,

	# Letters
	"A": 1,
	"B": 2,
	"C": 3,
	"D": 4,
	"E": 5,
	"F": 6,

	'not at all relevant': 1,
	'not very relevant': 2,
	'slightly relevant': 3,
	'somewhat relevant': 4,
	'very relevant': 5,
	'extremely relevant': 6,

	'strongly disagree': 1,
	'moderately disagree': 2,
	'slightly disagree': 3,
	'slightly agree': 4,
	'moderately agree': 5,
	'strongly agree': 6,
}

response_map_en_macq = {
	# Numeric strings
	"1": 1,
	"2": 2,
	"3": 3,
	"4": 4,
	"5": 5,
	"6": 6,

	# Integers 
	1: 1,
	2: 2,
	3: 3,
	4: 4,
	5: 5,
	6: 6,

	# Letters
	"A": 1,
	"B": 2,
	"C": 3,
	"D": 4,
	"E": 5,
	"F": 6,

	'not at all relevant': 1,
	'not very relevant': 2,
	'slightly relevant': 3,
	'somewhat relevant': 4,
	'very relevant': 5,
	'extremely relevant': 6,

	'strongly disagree': 1,
	'disagree': 2,
	'neither agree or disagree': 3,
	'agree': 4,
	'strongly agree': 5,
}

### Dictionaries with key groups ###
keys_mfq = {
	'care_rel': ['MFQ1_1', 'MFQ1_7', 'MFQ1_12'],
	'care_judg': ['MFQ2_1', 'MFQ2_7', 'MFQ2_12'],
	'fairness_rel': ['MFQ1_2', 'MFQ1_8', 'MFQ1_13'],
	'fairness_judg': ['MFQ2_2', 'MFQ2_8', 'MFQ2_13'],
	'loyalty_rel': ['MFQ1_3', 'MFQ1_9', 'MFQ1_14'],
	'loyalty_judg': ['MFQ2_3', 'MFQ2_9', 'MFQ2_14'],
	'authority_rel': ['MFQ1_4', 'MFQ1_10', 'MFQ1_15'],
	'authority_judg': ['MFQ2_4', 'MFQ2_10', 'MFQ2_15'],
	'purity_rel': ['MFQ1_5', 'MFQ1_11', 'MFQ1_16'],
	'purity_judg': ['MFQ2_5', 'MFQ2_11', 'MFQ2_16'],
	'care': ['MFQ1_1', 'MFQ1_7', 'MFQ1_12', 'MFQ2_1', 'MFQ2_7', 'MFQ2_12'],
	'fairness': ['MFQ1_2', 'MFQ1_8', 'MFQ1_13', 'MFQ2_2', 'MFQ2_8', 'MFQ2_13'],
	'loyalty': ['MFQ1_3', 'MFQ1_9', 'MFQ1_14', 'MFQ2_3', 'MFQ2_9', 'MFQ2_14'],
	'authority': ['MFQ1_4', 'MFQ1_10', 'MFQ1_15', 'MFQ2_4', 'MFQ2_10', 'MFQ2_15'],
	'purity': ['MFQ1_5', 'MFQ1_11', 'MFQ1_16', 'MFQ2_5', 'MFQ2_11', 'MFQ2_16'],
	'control': ['MFQ1_6', 'MFQ2_6']
}

keys_macq = {
	'family_rel': ['MACQ1_1', 'MACQ1_2', 'MACQ1_3'],
	'family_judg': ['MACQ2_1', 'MACQ2_2', 'MACQ2_3'],
	'group_rel': ['MACQ1_4', 'MACQ1_5', 'MACQ1_6'],
	'group_judg': ['MACQ2_4', 'MACQ2_5', 'MACQ2_6'],
	'reciprocity_rel': ['MACQ1_7', 'MACQ1_8', 'MACQ1_9'],
	'reciprocity_judg': ['MACQ2_7', 'MACQ2_8', 'MACQ2_9'],
	'heroism_rel': ['MACQ1_10', 'MACQ1_11', 'MACQ1_12'],
	'heroism_judg': ['MACQ2_10', 'MACQ2_11', 'MACQ2_12'],
	'deference_rel': ['MACQ1_13', 'MACQ1_14', 'MACQ1_15'],
	'deference_judg': ['MACQ2_13', 'MACQ2_14', 'MACQ2_15'],
	'fairness_rel': ['MACQ1_16', 'MACQ1_17', 'MACQ1_18'],
	'fairness_judg': ['MACQ2_16', 'MACQ2_17', 'MACQ2_18'],
	'property_rel': ['MACQ1_19', 'MACQ1_20', 'MACQ1_21'],
	'property_judg': ['MACQ2_19', 'MACQ2_20', 'MACQ2_21'],
	'family': ['MACQ1_1', 'MACQ1_2', 'MACQ1_3', 'MACQ2_1', 'MACQ2_2', 'MACQ2_3'],
	'group': ['MACQ1_4', 'MACQ1_5', 'MACQ1_6', 'MACQ2_4', 'MACQ2_5', 'MACQ2_6'],
	'reciprocity': ['MACQ1_7', 'MACQ1_8', 'MACQ1_9', 'MACQ2_7', 'MACQ2_8', 'MACQ2_9'],
	'heroism': ['MACQ1_10', 'MACQ1_11', 'MACQ1_12', 'MACQ2_10', 'MACQ2_11', 'MACQ2_12'],
	'deference': ['MACQ1_13', 'MACQ1_14', 'MACQ1_15', 'MACQ2_13', 'MACQ2_14', 'MACQ2_15'],
	'fairness': ['MACQ1_16', 'MACQ1_17', 'MACQ1_18', 'MACQ2_16', 'MACQ2_17', 'MACQ2_18'],
	'property': ['MACQ1_19', 'MACQ1_20', 'MACQ1_21', 'MACQ2_19', 'MACQ2_20', 'MACQ2_21']
}


### Columns ###
cols_mfq = [
	'MFQ1_1','MFQ1_2','MFQ1_3','MFQ1_4','MFQ1_5','MFQ1_6','MFQ1_7','MFQ1_8',
	'MFQ1_9','MFQ1_10','MFQ1_11','MFQ1_12','MFQ1_13','MFQ1_14','MFQ1_15','MFQ1_16',
	'MFQ2_1','MFQ2_2','MFQ2_3','MFQ2_4','MFQ2_5','MFQ2_6','MFQ2_7','MFQ2_8',
	'MFQ2_9','MFQ2_10','MFQ2_11','MFQ2_12','MFQ2_13','MFQ2_14','MFQ2_15','MFQ2_16'
]

cols_macq = [
	'MACQ1_1','MACQ1_2','MACQ1_3','MACQ1_4','MACQ1_5','MACQ1_6','MACQ1_7','MACQ1_8','MACQ1_9','MACQ1_10','MACQ1_11',
	'MACQ1_12','MACQ1_13','MACQ1_14','MACQ1_15','MACQ1_16','MACQ1_17','MACQ1_18','MACQ1_19','MACQ1_20','MACQ1_21',
	'MACQ2_1','MACQ2_2','MACQ2_3','MACQ2_4','MACQ2_5','MACQ2_6','MACQ2_7','MACQ2_8','MACQ2_9','MACQ2_10','MACQ2_11',
	'MACQ2_12','MACQ2_13','MACQ2_14','MACQ2_15','MACQ2_16','MACQ2_17','MACQ2_18','MACQ2_19','MACQ2_20','MACQ2_21'
]


##### Preparing the data #####

def convert_dist(df, cols):
	"""Only for scoring based df"""
	for col in cols:
		df[col] = df[col].apply(lambda x: [float(i) for i in re.sub(r'\[|\]', '', x).split(', ')])
	return df

	
def reverse_dist(df, gen=False, dist_cols=None):
	if gen:
		mask = (df['order']=='inverted') & (df['labels']!='labels_str')
		print('generation check')
	else:
		mask = df['order'] == 'inverted'
		print('scoring check')

	if df['quest'].unique() == 'macq':
		likert_cols1 = [col for col in cols_macq if 'MACQ1' in col]
		likert_cols2 = [col for col in cols_macq if 'MACQ2' in col]
		df.loc[mask, likert_cols1] = 7 - df.loc[mask, likert_cols1]
		df.loc[mask, likert_cols2] = 6 - df.loc[mask, likert_cols2]
	else:
		df.loc[mask, cols_mfq] = 7 - df.loc[mask, cols_mfq]

	if dist_cols is not None:
		for col in dist_cols:
			df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x[::-1])
	else:
		print('dist_cols not available or not reversed')

	return df

def reverse_macq(df):
	df[['MACQ2_19', 'MACQ2_20', 'MACQ2_21']] = df[['MACQ2_19', 'MACQ2_20', 'MACQ2_21']].replace({5:1, 4:2, 3:3, 2:4, 1:5})
	try:
		dist_cols = ['MACQ2_19_dist', 'MACQ2_19_prob', 'MACQ2_20_dist', 'MACQ2_20_prob', 'MACQ2_21_dist', 'MACQ2_21_prob']
		df[dist_cols] = df[dist_cols].apply(lambda x: x[::-1])
	except Exception as e:
		print(f'No dist_columns in df, {e}') 
	return df

def reverse_mfq(df):
	df['MFQ2_6'] = df['MFQ2_6'].replace({6:1,5:2,4:3,3:4,2:5,1:6})
	try:
		df[['MFQ2_6_dist', 'MFQ2_6_prob']] = df[['MFQ2_6_dist', 'MFQ2_6_prob']].apply(lambda x: x[::-1])
	except Exception as e:
		print(f'No dist_columns in df, {e}')
	return df

def bin_macq(df):
	for col in cols_macq:
		if 'MACQ1' in col:
			df[col] = pd.cut(df[col],
				bins=[0, 10, 30, 50, 70, 90, 100],
				labels=False,
				include_lowest=True) + 1
		else:
			df[col] = pd.cut(df[col],
				bins=[0, 12.5, 37.5, 62.5, 87.5, 100],
				labels=False,
				include_lowest=True) + 1
	return df

def get_freq(df, columns):
	rel_freq = {}
	for col in columns:
		rel_freq[f'{col}_prob'] = list(df[col].value_counts(normalize = True).sort_index())
	return rel_freq

### For generated data ###
def extract_matches(response, scale_name, label_type, language='nor'):
	"""
	Returns a list of all matching answers found in the response.
	"""

	if pd.isna(response):
		return []

	text = str(response)

	if label_type == 'labels_str':
		# Case-insensitive search for answer strings
		matches = []
		lower = text.lower()

		# Search longest phrases first 
		if language == 'en':
			options = sorted(scales_en[scale_name], key=len, reverse=True)
		else:
			options = sorted(scales[scale_name], key=len, reverse=True)

		for option in options:
			if option.lower() in lower:
				matches.append(option)
				# Remove the matched phrase so shorter substrings
				# (e.g. "enig" in "svært enig") are not found
				lower = lower.replace(option.lower(), ' ')

		return matches

	elif label_type == 'labels_num':
		n = len(scales[scale_name])
		pattern = rf'\b([1-{n}])\b'
		return re.findall(pattern, text)

	elif label_type == 'labels_alpha':
		last = chr(ord('A') + len(scales[scale_name]) - 1)
		pattern = rf'\b([A-{last}])\b'
		return re.findall(pattern, text)

	else:
		return []


def parse_results(df, language='nor'):
	# All response columns
	response_cols = [c for c in df.columns if c.endswith('_response')]
	
	# Create parsed columns
	for col in response_cols:
	
		# Determine scale
		if col.startswith('MFQ1'):
			scale = 'MFQ1'
		elif col.startswith('MFQ2'):
			scale = 'MFQ2'
		elif col.startswith("MACQ1"):
			scale = 'MACQ1'
		elif col.startswith('MACQ2'):
			scale = 'MACQ2'
		else:
			continue
	
		parsed_col = col.replace('_response', '')
	
		df[parsed_col] = [
			extract_matches(resp, scale, labels, language)
			for resp, labels in zip(df[col], df['labels'])
		]
	return df

def recode_list(lst, mapping):
	out = []
	for x in lst:
		if x in mapping:
			out.append(mapping[x])
		else:
			raise ValueError(f'Unknown response: {x!r} in column {col}')   
	return out

def resolve_response(lst, quest = 'mfq'):
	"""
	Extra information on response behavior under generation
	Only refers to one item at a time!
	Returns:
		value: resolved response (if resolvable, otherwise nan)
		empty: 1 if empty response
		recoded: 1 if multiple values on same side of scale were recoded (e.g., [1, 3] were parsed)
		contradiction: 1 if contradictory responses (e.g., [2, 5] were parsed)
	"""

	# Empty list
	if len(lst) == 0:
		return np.nan, 1, 0, 0

	# One response
	if len(lst) == 1:
		return lst[0], 0, 0, 0

	# Remove duplicates
	responses = sorted(set(lst))

	# All identical
	if len(responses) == 1:
		return responses[0], 0, 0, 0

	if quest == 'macq2': # Only 5-point Likert scale
		# Same side: 1-2; resolves to uenig 
		if all(r <= 2 for r in responses):
			return 2, 0, 1, 0

		# Same side: 4-5; resolves to enig
		if all(r >= 4 for r in responses):
			return 4, 0, 1, 0

	else:
		# Same side: 1-3
		if all(r <= 3 for r in responses):
			return 2, 0, 1, 0

		# Same side: 4-6
		if all(r >= 4 for r in responses):
			return 5, 0, 1, 0

	# Contradiction
	return np.nan, 0, 0, 1


def scores(df, keys):
	"""Determine scores for foundations/domains for content analysis
	Only valid for content analysis after recoding inverted items"""

	for key in keys.keys():
		df[key] = df[keys[key]].mean(axis=1)
	return df

# Response behavior metrics
def repetition(x):
	"""Proportion of exact matches between consecutive responses."""
	x = np.asarray(x)
	return np.mean(x[:-1] == x[1:])


def counting_detected(x):
	"""
	Detects whether the row contains counting (e.g., 1,2,3,4,5 or 5,4,3,2,1).
	"""
	x = np.asarray(x)

	# Only considering up to 5 to be universal across scales (incl. MACQ2)
	asc = np.array([1, 2, 3, 4, 5])
	desc = asc[::-1]

	for i in range(len(x) - 4):
		window = x[i:i+5]
		if np.array_equal(window, asc) or np.array_equal(window, desc):
			return 1

	return 0



def response_behavior(df, cols):
	responses = df[cols]
	df['counting_detected'] = responses.apply(counting_detected, axis=1)
	df['repetition'] = responses.apply(repetition, axis=1)
	df['n_unique'] = responses.nunique(axis=1)

	return df

def response_behavior_div(df, cols_1 = cols_mfq[:16], cols_2 = cols_mfq[16:]):
	"""Response behavior divided by scale (Relevance vs Judgment)"""
	df['counting_detected_rel'] = df[cols_1].apply(counting_detected, axis=1)
	df['counting_detected_judg'] = df[cols_2].apply(counting_detected, axis=1)
	df['repetition_rel'] = df[cols_1].apply(repetition, axis=1)
	df['repetition_judg'] = df[cols_2].apply(repetition, axis=1)
	df['n_unique_rel'] = df[cols_1].nunique(axis=1)
	df['n_unique_judg'] = df[cols_2].nunique(axis=1)
	df['sd_response_rel'] = df[cols_1].std(axis=1)
	df['sd_response_judg'] = df[cols_2].std(axis=1)
	df['iqr_rel'] = df[cols_1].apply(iqr, axis=1)
	df['iqr_judg'] = df[cols_2].apply(iqr, axis=1)

	return df


def prepare_llm(df, content = False):
	"""Preparing llm scoring responses
	content: content analysis or response behavior, if False no recoding of control or reversed items (default = False)"""
	df[cols_mfq+cols_macq] = df[cols_mfq+cols_macq] + 1

	macq = df[df['quest'] == 'macq'].dropna(axis=1)
	mfq = df[df['quest'] == 'mfq'].dropna(axis=1)

	cols = [col for col in macq.columns if 'dist' in col or 'prob' in col]
	macq = convert_dist(macq, cols)
	# macq = reverse_dist(macq, cols) # debatable
	macq['setting'] = np.select(
		[macq['inst'], macq['cloze']],['inst', 'cloze'],
		default='basic'
	) 
	if content: 
		# Content analysis
		macq = reverse_dist(macq, gen=False, dist_cols=cols)
		macq = reverse_macq(macq)
		macq = scores(macq, keys_macq)
	else: 
		# Response behavior analysis 
		macq = response_behavior(macq, cols_macq)
		macq = response_behavior_div(macq, cols_macq[:21], cols_macq[21:])


	cols = [col for col in mfq.columns if 'dist' in col or 'prob' in col]
	mfq = convert_dist(mfq, cols)
	mfq['setting'] = np.select(
		[mfq['inst'], mfq['cloze']],['inst', 'cloze'],
		default='basic'
	)
	if content:
		mfq = reverse_dist(mfq, gen=False, dist_cols=cols)
		mfq = scores(mfq, keys_mfq)
	else:
		mfq = response_behavior(mfq, cols_mfq)
		mfq = response_behavior_div(mfq, cols_mfq[:16], cols_mfq[16:])
	
	return macq, mfq

def prepare_llm_gen(df, content = False, language = 'nor'):
	"""Preparing generation-based responses"""
	df = parse_results(df, language)
	df['parse_error'] = df[cols_mfq + cols_macq].apply(lambda row: any(len(lst) > 1 for lst in row), axis=1).astype(int)
	if language == 'en':
		for col in cols_mfq:
			df[col] = df[col].apply(lambda x: recode_list(x, response_map_en_mfq))
		for col in cols_macq:
			df[col] = df[col].apply(lambda x: recode_list(x, response_map_en_macq))
	else:
		for col in cols_mfq+cols_macq:
			df[col] = df[col].apply(lambda x: recode_list(x, response_map_nor))

	mfq = df[df['quest'] == 'mfq']
	macq = df[df['quest'] == 'macq']
	mfq = mfq.dropna(axis=1, how='all')
	macq = macq.dropna(axis=1, how='all')

	# Initialize counters
	mfq['n_empty'] = 0
	mfq['n_recoded'] = 0
	mfq['n_contradictions'] = 0
	macq['n_empty'] = 0
	macq['n_recoded'] = 0
	macq['n_contradictions'] = 0

	for col in cols_mfq:
		results = mfq[col].apply(resolve_response)

		mfq[col] = results.str[0]
		mfq['n_empty'] += results.str[1]
		mfq['n_recoded'] += results.str[2]
		mfq['n_contradictions'] += results.str[3]

	mfq['setting'] = np.select(
		[mfq['inst'], mfq['cloze']],['inst', 'cloze'],
		default='basic'
	)

	for col in cols_macq:
		results = macq[col].apply(resolve_response, quest = 'macq2') if col in cols_macq[21:] else macq[col].apply(resolve_response)

		macq[col] = results.str[0]
		macq['n_empty'] += results.str[1]
		macq['n_recoded'] += results.str[2]
		macq['n_contradictions'] += results.str[3]

	macq['setting'] = np.select(
		[macq['inst'], macq['cloze']],['inst', 'cloze'],
		default='basic'
	)

	if content:
		mfq = reverse_dist(mfq, gen=True)
		mfq = scores(mfq, keys_mfq)

		macq = reverse_dist(macq, gen=True)
		macq = reverse_macq(macq)
		macq = scores(macq, keys_macq)

	else: 
		# Response behavior analysis 
		macq = response_behavior(macq, cols_macq)
		macq = response_behavior_div(macq, cols_macq[:21], cols_macq[21:])
		mfq = response_behavior(mfq, cols_mfq)
		mfq = response_behavior_div(mfq, cols_mfq[:16], cols_mfq[16:])

	return macq, mfq


def prepare_human(df, content = False):
	"""Prepare human sample"""
	df['model'] = 'human'
	df['scoring'] = 'human'
	macq = df[df['FILTER1'] == 1].dropna(axis=1)
	macq['quest'] = 'macq'
	mfq = df[df['FILTER1'] == 2].dropna(axis=1)
	mfq['quest'] = 'mfq'

	macq = bin_macq(macq)
	macq_freq = get_freq(macq, [col for col in macq.columns if 'MACQ' in col])
	mfq_freq = get_freq(mfq, [col for col in mfq.columns if 'MFQ' in col])
	if content:
		macq = reverse_macq(macq)
		macq = scores(macq, keys_macq)
		mfq = scores(mfq, keys_mfq)
	else: 
		# Response behavior analysis 
		macq = response_behavior(macq, cols_macq)
		macq = response_behavior_div(macq, cols_macq[:21], cols_macq[21:])
		mfq = response_behavior(mfq, cols_mfq)
		mfq = response_behavior_div(mfq, cols_mfq[:16], cols_mfq[16:])

	return macq, macq_freq, mfq, mfq_freq


##### Content Analysis #####
### Descriptive overview ###
def overview_llm(df, plot = False):
	"""Provide descriptive overview of content columns
	Only apply with prepared dataset content = True!
	plot = True will provide histograms for all content columns"""
	if df['quest'].unique() == 'mfq':
		cols = cols_mfq
	else: 
		cols = cols_macq
	print('\n__________________MODEL___________________\n')
	print('\n------------------Mean--------------------\n')
	display(df.groupby('model')[cols].mean())
	print('\n------------------Standard deviation--------------------\n')
	display(df.groupby('model')[cols].std())
	print('\n------------------Range--------------------\n')
	display(df.groupby('model')[cols].apply(lambda x: x.max() - x.min()))
	if plot:
		for model in df['model'].unique():
			for col in cols:
				plt.figure(figsize=(5, 3))
				sns.countplot(data=df[df['model'] == model], x=col)
				plt.title(f"{model} - {col}")
				plt.show()

	print('\n__________________LABELS___________________\n')
	print('\n------------------Mean--------------------\n')
	display(df.groupby('labels')[cols].mean())
	print('\n------------------Standard deviation--------------------\n')
	display(df.groupby('labels')[cols].std())
	print('\n------------------Range--------------------\n')
	display(df.groupby('labels')[cols].apply(lambda x: x.max() - x.min()))

	print('\n__________________CONV_TYPE___________________\n')
	print('\n------------------Mean--------------------\n')
	display(df.groupby('conv_type')[cols].mean())
	print('\n------------------Standard deviation--------------------\n')
	display(df.groupby('conv_type')[cols].std())
	print('\n------------------Range--------------------\n')
	display(df.groupby('conv_type')[cols].apply(lambda x: x.max() - x.min()))

	print('\n__________________ORDER___________________\n')
	print('\n------------------Mean--------------------\n')
	display(df.groupby('order')[cols].mean())
	print('\n------------------Standard deviation--------------------\n')
	display(df.groupby('order')[cols].std())
	print('\n------------------Range--------------------\n')
	display(df.groupby('order')[cols].apply(lambda x: x.max() - x.min()))

	print('\n__________________SETTING___________________\n')
	print('\n------------------Mean--------------------\n')
	display(df.groupby('setting')[cols].mean())
	print('\n------------------Standard deviation--------------------\n')
	display(df.groupby('setting')[cols].std())
	print('\n------------------Range--------------------\n')
	display(df.groupby('setting')[cols].apply(lambda x: x.max() - x.min()))

def overview_human(df, cols = cols_mfq, plot = False):
	print('\n__________________HUMAN___________________\n')
	print('\n------------------Mean--------------------\n')
	display(df[cols].mean())
	print('\n------------------Standard deviation--------------------\n')
	display(df[cols].std())
	print('\n------------------Range--------------------\n')
	display(df[cols].apply(lambda x: x.max() - x.min()))
	if plot:
		for col in cols:
			plt.figure(figsize=(5, 3))
			sns.countplot(data=df, x=col)
			plt.title(f"{col}")
			plt.show()


##### Analysis of response behavior #####
### Kmeans clustering ###

def prepare_cluster(df, features = ['counting_detected', 'repetition', 'n_unique']):
	# Features for clustering response behavior (define out of response behavior features)
	X = df[features].copy()
	
	# Standardize the features
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	return X_scaled

def find_k(X_scaled, random_state=8, n_init=10, filename=None):
	### Find number of clusters via elbow method and silhouette analysis ###
	inertia = []
	silhouette_scores = []
	K = range(1, 11)
	
	for k in K:
		kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
		kmeans.fit(X_scaled)
		inertia.append(kmeans.inertia_)
		if k > 1:
			labels = kmeans.predict(X_scaled)
			silhouette_scores.append(silhouette_score(X_scaled, labels))
	
	plt.figure(figsize=(6, 4))
	
	plt.plot(K, inertia, marker="o")
	plt.xlabel("Number of clusters (k)")
	plt.ylabel("Within-cluster sum of squares (Inertia)")
	plt.title("Elbow Method")
	plt.xticks(K)
	
	if filename is not None:
		plt.savefig(f"figures/kmeans/elbow_{filename}.pdf")
	else:
		plt.show()
	
	plt.figure(figsize=(6, 4))
	
	plt.plot(range(2, 11), silhouette_scores, marker="o")
	plt.xlabel("Number of clusters (k)")
	plt.ylabel("Silhouette score")
	plt.title("Silhouette Analysis")
	
	if filename is not None:
		plt.savefig(f"figures/kmeans/silhouette_{filename}.pdf")
	else:
		plt.show()

def kmeans(X_scaled, df, n_clusters, random_state=8):
	"""Applies kmeans directly"""
	cluster = KMeans(n_clusters, random_state=random_state) 
	df['kmeans_cluster'] = cluster.fit_predict(X_scaled)
	return df

def plot_kmeans(X_scaled, df, style='model', legend_pos='upper left', filename=None):
	"""Plot clusters, PCA for plotting purposes"""
	pca = PCA(n_components=2)
	coords = pca.fit_transform(X_scaled)
	
	df['PC1'] = coords[:, 0]
	df['PC2'] = coords[:, 1]
	
	sns.scatterplot(
		data=df,
		x='PC1',
		y='PC2',
		hue='kmeans_cluster',
		style=style
	)
	plt.legend(
		fontsize=6,
		title_fontsize=9,
		markerscale=0.7,
		loc = legend_pos
	)
	
	if filename is not None:
		plt.savefig(f'figures/kmeans/{filename}.pdf', bbox_inches='tight')
	else:
		plt.show()


##### Outlier detection using DBSCAN (following Huang et al. 2024) #####
def dbscan(df, cond, eps, min_samples):
	"""DBSCAN applied to PCA transformed data!"""
	if cond == 'mfq':
		features = [
			'care',
			'fairness',
			'loyalty',
			'authority',
			'purity'
		]
	elif cond == 'macq':
		features = [
			'family',
			'group',
			'reciprocity',
			'heroism',
			'deference',
			'fairness',
			'property'
		]
	elif cond == 'response_behavior':
		features = [
			'repetition',
			'n_unique',
			'counting_detected',
			'iqr'
			# 'sd_response'
		]
	
	X = df[features].copy()
	
	# Standardize the features
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	pca = PCA(n_components=2)
	coords = pca.fit_transform(X_scaled)
	
	df['PC1'] = coords[:, 0]
	df['PC2'] = coords[:, 1]

	dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)

	df['dbscan'] = dbscan 								# Actual clusters
	df['dbscan_out'] = np.where(dbscan == -1, 1, 0)		# 1: Outliers

	return df

def plot_dbscan(df, style='model', filename=None):
	sns.scatterplot(
		data=df,
		x='PC1',
		y='PC2',
		hue='dbscan_out',
		style=style
	)
	plt.legend(
		fontsize=6,
		title_fontsize=9,
		markerscale=0.7,
		loc = 'upper left'
	)
	
	if filename is not None:
		plt.savefig(f'figures/dbscan/{filename}.pdf', bbox_inches='tight')
	else:
		plt.show()


##### Comparison between distributions #####
### Mahalanobis distance (rank-transformed): comparing one obs to the mean of a distribution considering covariance in said distribution ###
def mahalanobis_distance_ordinal(
    target_dist,
    ref_dist,
    features,
    categories,
    return_details=False
):
    """
    Compute Mahalanobis distances for ordinal response data based on midpoint ranks

    target_dist: pd.DataFrame

    ref_dist: pd.DataFrame

    features: sequence of str
        Ordinal columns to include.

    categories: sequence
        Ordered response categories, from lowest to highest. The same category
        order is assumed for every feature.

    return_details: bool, default=False
        If True, return distances plus the reference-based ordinal encoding
        and covariance information.

    Returns:
    np.ndarray or dict
        One Mahalanobis distance per row in target_dist. Rows with missing
        target values receive np.nan.
    """
    if not isinstance(target_dist, pd.DataFrame):
        raise TypeError("target_dist must be a pandas DataFrame.")

    if not isinstance(ref_dist, pd.DataFrame):
        raise TypeError("ref_dist must be a pandas DataFrame.")

    features = list(features)
    categories = list(categories)

    if not features:
        raise ValueError("At least one feature is required.")

    if len(categories) < 2:
        raise ValueError("At least two ordered categories are required.")

    if len(set(categories)) != len(categories):
        raise ValueError("categories must not contain duplicates.")

    missing_features = (
        set(features)
        - (set(target_dist.columns) & set(ref_dist.columns))
    )

    if missing_features:
        raise ValueError(
            f"Features not found in both data frames: {sorted(missing_features)}"
        )

    def validate_values(df, dataset_name):
        for feature in features:
            non_missing = df[feature].dropna()
            invalid = non_missing[~non_missing.isin(categories)].unique()

            if len(invalid) > 0:
                raise ValueError(
                    f"Invalid values in {dataset_name}[{feature!r}]: "
                    f"{invalid.tolist()}. "
                    f"All non-missing values must appear in categories."
                )

    validate_values(ref_dist, "ref_dist")
    validate_values(target_dist, "target_dist")

    # Remove incomplete reference rows because mean/covariance estimation
    # requires complete feature vectors.
    ref_complete = ref_dist[features].dropna()

    if len(ref_complete) < 2:
        raise ValueError(
            "At least two complete reference observations are required."
        )

    ordinal_scores = {}
    ref_encoded = pd.DataFrame(index=ref_complete.index)
    target_encoded = pd.DataFrame(index=target_dist.index)

    for feature in features:
        # Marginal frequencies estimated only from the reference sample
        counts = (
            ref_complete[feature]
            .value_counts()
            .reindex(categories, fill_value=0)
            .to_numpy(dtype=float)
        )

        total = counts.sum()

        if total == 0:
            raise ValueError(
                f"No usable reference ratings found for feature {feature!r}."
            )

        midpoint = (np.cumsum(counts) - counts / 2.0) / total

        score_map = {
            category: midpoint[i]
            for i, category in enumerate(categories)
        }

        ordinal_scores[feature] = score_map
        ref_encoded[feature] = ref_complete[feature].map(score_map)
        target_encoded[feature] = target_dist[feature].map(score_map)

    # Remove features without variation before covariance estimation.
    varying_features = [
        feature
        for feature in features
        if ref_encoded[feature].nunique() > 1
    ]

    if not varying_features:
        raise ValueError(
            "None of the ordinal features vary in the reference distribution."
        )

    ref_encoded = ref_encoded[varying_features]
    target_encoded = target_encoded[varying_features]

    mean_ref = ref_encoded.mean(axis=0).to_numpy()
    cov = LedoitWolf().fit(ref_encoded) # LedoitWolf covariance since num_features might be larger than num_observations
    cov_matrix = cov.covariance_

    cov_matrix = np.atleast_2d(cov_matrix)

    # pseudoinverse reference cov matrix instead --> supposed to be more stable than inv for ordinal data
    inv_cov_matrix = np.linalg.pinv(cov_matrix)

    distances = np.full(len(target_encoded), np.nan, dtype=float)

    complete_target = target_encoded.notna().all(axis=1)

    X_target = target_encoded.loc[
        complete_target,
        varying_features
    ].to_numpy()

    distances[complete_target.to_numpy()] = np.array([
        mahalanobis(x, mean_ref, inv_cov_matrix)
        for x in X_target
    ])

    # Preserve the original target_df index.
    distances = pd.Series(distances, index=target_dist.index, name="distance")

    if return_details:
        return {
            "distances": distances,
            "ordinal_scores": ordinal_scores,
            "features_used": varying_features,
            "reference_mean": mean_ref,
            "reference_covariance": cov_matrix,
            "reference_n_complete": int(len(ref_encoded)),
            "target_n_complete": int(complete_target.sum()),
        }

    return distances

def add_groupwise_ordinal_mahalanobis(
    df,
    group_col,
    features,
    categories,
    distance_col="mahalanobis_distance_ordinal"
):
    """
    Leave-one-out ordinal Mahalanobis distance per group (e.g., model).

    For every row:
        target_dist = that one row
        ref_dist = all other rows in the same `group_col` subgroup

    df: pd.DataFrame
        Full data set.

    group_col: str
        Column defining the subgroups, e.g., 'model'

    features: sequence of str

    categories: sequence

    distance_col : str, Name of the new distance column.

	Returns:
    pd.DataFrame
        Copy of df with `distance_col` added.
    """

    result = df.copy()
    distances = np.full(len(result), np.nan, dtype=float)

    # Resetting for safe positional assignment, including if df has
    # duplicate index labels.
    work = result.reset_index(drop=True)

    for group_value, positions in work.groupby(
        group_col,
        dropna=False,
        sort=False
    ).indices.items():

        positions = np.asarray(positions)

        # For each row in this subgroup:
        for target_position in positions:

            # Rest of the rows of that subgroup form the reference sample.
            reference_positions = positions[positions != target_position]

            # Need at least 2 reference observations for covariance estimation.
            if len(reference_positions) < 2:
                continue

            target_row = work.iloc[[target_position]]
            reference_rows = work.iloc[reference_positions]

            try:
                distance = mahalanobis_distance_ordinal(
                    target_dist=target_row,
                    ref_dist=reference_rows,
                    features=features,
                    categories=categories
                )

                # The function returns a one-element Series for one target row.
                distances[target_position] = distance.iloc[0]

            except ValueError:
                # Too few complete reference rows, no feature
                # variation in the reference group, or invalid data.
                distances[target_position] = np.nan

    result[distance_col] = distances

    return result




##### Krippendorff's alpha with ordinal metric difference function (rank-transformed) #####
def krippendorff_alpha_ordinal(ratings, categories, return_details=False):
    """
    ratings: array-like, shape (n_items, n_judges)
        One row per item and one column per judge. Missing values may be
        None, np.nan, or pd.NA.

    categories: sequence
        Ordinal categories in increasing order, for example:
        ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]

    return_details: bool, default=False
        If True, return a dictionary with alpha, observed disagreement,
        expected disagreement, and sample information.

    Returns:
    float or dict
        Krippendorff's alpha, or diagnostic details if return_details=True.

    - Items with fewer than 2 non-missing ratings are excluded.
    - ValueError if alpha is undefined when every usable rating belongs to one category.
    """
    ratings = np.asarray(ratings, dtype=object)

    if ratings.ndim != 2:
        raise ValueError("ratings must be a 2D array-like object.")

    categories = list(categories)
    if len(categories) < 2:
        raise ValueError("At least two ordered categories are required.")

    if len(set(categories)) != len(categories):
        raise ValueError("categories must not contain duplicates.")

    category_index = {category: i for i, category in enumerate(categories)}
    n_categories = len(categories)

    def is_missing(value):
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    # Build category-count vectors for each usable item.
    # An item is usable only if at least 2 judges rated it.
    item_counts = []

    for row in ratings:
        counts = np.zeros(n_categories, dtype=float)

        for value in row:
            if is_missing(value):
                continue

            if value not in category_index:
                raise ValueError(
                    f"Rating {value!r} is not present in categories={categories!r}"
                )

            counts[category_index[value]] += 1

        if counts.sum() >= 2:
            item_counts.append(counts)

    if not item_counts:
        raise ValueError("No items have at least two non-missing ratings.")

    item_counts = np.vstack(item_counts)

    # Marginal category frequencies among usable ratings.
    marginal = item_counts.sum(axis=0)
    total_ratings = marginal.sum()

    if total_ratings < 2:
        raise ValueError("Fewer than two usable ratings remain.")

   
    midpoint = np.cumsum(marginal) - marginal / 2.0

    distance = (midpoint[:, None] - midpoint[None, :]) ** 2
    
    observed_numerator = 0.0

    for counts in item_counts:
        n = counts.sum()
        observed_numerator += (counts @ distance @ counts) / (n - 1.0)

    observed_disagreement = observed_numerator / total_ratings

    # Expected disagreement under chance agreement.
    expected_disagreement = (
        marginal @ distance @ marginal
    ) / (total_ratings * (total_ratings - 1.0))

    if np.isclose(expected_disagreement, 0.0):
        raise ValueError(
            "Expected disagreement is zero; alpha is undefined. "
            "All usable ratings appear to belong to the same category."
        )

    alpha = 1.0 - observed_disagreement / expected_disagreement

    if return_details:
        return {
            "alpha": float(alpha),
            "observed_disagreement": float(observed_disagreement),
            "expected_disagreement": float(expected_disagreement),
            "usable_items": int(len(item_counts)),
            "usable_ratings": int(total_ratings),
            "marginal_category_counts": dict(zip(categories, marginal.astype(int))),
        }

    return float(alpha)


### Logistic regression with categorical predictors ###
def fit_logit_with_categoricals(
    df,
    outcome,
    features=None,
    robust_se="HC0",
    drop_missing=True,
):
    """
    Fit a logistic regression with automatic handling of categorical predictors.

    df: pd.DataFrame
    outcome: str, Name of the binary outcome column 
    features: list[str], Predictor columns. If None, uses every column except `outcome`.
    robust_se : str or None, Robust standard error type
    drop_missing : bool
        If True, drop rows with missing values in model variables.

    Returns:
	    model: statsmodels discrete-model results object
	    model_df: pd.DataFrame
	        Data actually used for fitting.
	    formula: str
	        Formula used in the regression.
    """

    if outcome not in df.columns:
        raise ValueError(f"Outcome column '{outcome}' is not in df.")

    if features is None:
        features = [col for col in df.columns if col != outcome]

    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(f"These features are missing from df: {missing_features}")

    model_df = df[[outcome] + features].copy()

    y = model_df[outcome]
    if pd.api.types.is_bool_dtype(y):
        model_df[outcome] = y.astype(int)
    else:
        levels = y.dropna().unique()
        if len(levels) != 2:
            raise ValueError(
                f"Outcome must have exactly 2 distinct values, found {len(levels)}: {levels}"
            )
        if set(levels) != {0, 1}:
            lo, hi = sorted(levels)
            print(f"Recoding outcome: {lo!r} -> 0, {hi!r} -> 1")
            model_df[outcome] = y.map({lo: 0, hi: 1})
        model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")

    # Identify categorical predictors
    categorical_cols = [
        col for col in features
        if (
            pd.api.types.is_object_dtype(model_df[col])
            or isinstance(model_df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(model_df[col])
            or pd.api.types.is_bool_dtype(model_df[col])
        )
    ]
    numeric_cols = [col for col in features if col not in categorical_cols]

    kept_features = numeric_cols + categorical_cols
    model_df = model_df[[outcome] + kept_features]

    for col in categorical_cols:
        model_df[col] = model_df[col].astype("category")

    if drop_missing:
        initial_n = len(model_df)
        model_df = model_df.dropna()
        print(f"Rows retained after dropping missing values: {len(model_df):,} / {initial_n:,}")

    if len(model_df) == 0:
        raise ValueError("No rows remain after preprocessing.")

    # Check class balance
    counts = model_df[outcome].value_counts().sort_index()
    print("\nOutcome distribution:")
    print(counts.to_string())
    n_params_approx = len(numeric_cols) + len(categorical_cols)
    if counts.min() < 10 * n_params_approx:
        print(
            f"Warning: minority class has {counts.min()} cases for ~{n_params_approx}+ predictors "
            "(rule of thumb: >=10 events per predictor parameter)."
        )

    def quoted(col):
        return f'Q("{col}")'

    formula_terms = (
        [quoted(col) for col in numeric_cols]
        + [f"C({quoted(col)})" for col in categorical_cols]
    )
    formula = f"{quoted(outcome)} ~ " + " + ".join(formula_terms)

    logit = smf.logit(formula=formula, data=model_df)
    if robust_se is not None:
        model = logit.fit(disp=0, cov_type=robust_se)
    else:
        model = logit.fit(disp=0)

    if not model.mle_retvals.get("converged", True):
        print("Warning: optimizer did not converge (possible separation or collinearity).")

    print("\nFormula:")
    print(formula)
    print("\nCategorical predictors:")
    print(categorical_cols)
    print("\nNumeric predictors:")
    print(numeric_cols)

    print("\nLogit results:")
    print(model.summary())

    # Odds ratios with 95% CI
    ci = model.conf_int()
    odds = pd.DataFrame({
        "odds_ratio": np.exp(model.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": model.pvalues,
    })
    print("\nOdds ratios (95% CI):")
    print(odds.round(4).to_string())

    # Likelihood-ratio test against the null model
    print(f"\nMcFadden pseudo R2: {model.prsquared:.4f}")
    print(f"LR test: chi2 = {model.llr:.2f}, p = {model.llr_pvalue:.4g}")

    return model, model_df, formula


