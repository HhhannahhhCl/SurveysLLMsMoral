# Survey Experiments on LLMs - MFT \& MAC
This repository gathers necessary materials, questionnaires and code for conducting experiments described in "Neither Agree Nor Disagree: Reliability Between and Within LLM Moral Survey Responses" (Anonymous, 2026)

## Code
* read_data.py contains the function to read the questionnaires into a dict with a structured format.
* generate.py contains the necessary functions for scoring-based inference
* generate_batched.py contains the equivalent functions applied in batches for generation-based inference
* respond_all.py executes the workflow depending on the arguments generate and decoding
* creat_df.py loads results as json files and concatenates them to a pd.DataFrame

### Tokenization
Some models tokenize the string '\n' differently depending on the position in the sequence leading to manual hardcoding of the respective token in these cases. 

## Data 
The respective questionnaires are part of this repository in the respective folder. In order to compare LLM to human responses, an existing dataset representing a sample from the Norwegian adult population will be used (Enstad \& Finseraas, 2023). The dataset is made available by the Insititutt for samfunnsforskning ([Moralske intuisjoner og politiske orienteringer blant norske velgere](https://doi.org/10.18712/NSD-NSD3132-V1)). The sample contains a total of 2562 cases and is randomly split between the MFQ-30 (Graham et al., 2011) and MACQ (Curry et al., 2019).

| **Questionnaire** | **N**|
|-------------------|------|
| MFQ-30 | 1282 |
| MACQ | 1280 |
| Total | 2562 |

### Questionnaires
Each questionnaire has two versions. \_en contains the original English questionnaire. \_nor contains the Norwegian questionnaire presented to respondents. The questionnaires contain the adjustments described in the paper to ensure improve comparability.

## References
Curry, O. S., Mullins, D. A., & Whitehouse, H. (2019). Is it good to cooperate?: Testing the theory of morality-as-cooperation in 60 societies. Current Anthropology, 60(1), 47-69. doi: 10.1086/701478.

Enstad, J. D., & Finseraas, H. (2023). Moralske intuisjoner og politiske orienteringer blant norske velgere. Sikt - Kunnskapssektorens tjenesteleverandør. doi: 10.18712/NSD-NSD3132-V1.

Graham, J., Nosek, B. A., Haidt, J., Iyer, R., Koleva, S., & Ditto, P. H. (2011). Mapping the Moral Domain. Journal of Personality and Social Psychology, 101(2), 366–385.

