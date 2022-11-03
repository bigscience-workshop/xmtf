# !pip install -q datasets
# !pip install -q python-Levenshtein

from datasets import load_dataset
from Levenshtein import distance as levenshtein_distance


langs = ["th", "tr", "el"]

def compute_lev(sample):
    sample["levenshtein"] = levenshtein_distance(sample["premise"], sample["hypothesis"])
    return sample

for lang in langs:
    print("Language:", lang)
    xnli_th = load_dataset("xnli", lang)

    xnli_th_val = xnli_th["validation"]
    xnli_th_val_lev = xnli_th_val.map(compute_lev)

    xnli_th_val_lev_laba = xnli_th_val_lev.filter(lambda x: x["label"] == 0)
    xnli_th_val_lev_labb = xnli_th_val_lev.filter(lambda x: x["label"] == 1)
    xnli_th_val_lev_labc = xnli_th_val_lev.filter(lambda x: x["label"] == 2)

    laba_avg = sum(xnli_th_val_lev_laba["levenshtein"]) / len(xnli_th_val_lev_laba["levenshtein"])
    labb_avg = sum(xnli_th_val_lev_labb["levenshtein"]) / len(xnli_th_val_lev_labb["levenshtein"])
    labc_avg = sum(xnli_th_val_lev_labc["levenshtein"]) / len(xnli_th_val_lev_labc["levenshtein"])

    assert len(xnli_th_val_lev_laba) == len(xnli_th_val_lev_labb) == len(xnli_th_val_lev_labc)

    print("Entailment: ", laba_avg)
    print("Neutral: ", labb_avg)
    print("Contradiction: ", labc_avg)
    print("Samples: ", len(xnli_th_val_lev_labc))
    print("-"*50)

# Output:
"""
Language: th
WARNING:datasets.builder:Found cached dataset xnli (/root/.cache/huggingface/datasets/xnli/th/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd)
100%
3/3 [00:00<00:00, 57.44it/s]
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/th/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-5936ac8dd6e492bf.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/th/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-166b8840e7ce693d.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/th/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-fa931d7605599d3f.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/th/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-81b851032ab1083e.arrow
Entailment:  79.07710843373494
Neutral:  82.63734939759036
Contradiction:  81.51807228915662
Samples:  830
--------------------------------------------------
Language: tr
WARNING:datasets.builder:Found cached dataset xnli (/root/.cache/huggingface/datasets/xnli/tr/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd)
100%
3/3 [00:00<00:00, 63.90it/s]
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/tr/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-83a0fd0b8e2cfe5d.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/tr/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-f83874828d44e3e4.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/tr/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-74f594889e89abd6.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/tr/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-7211cfac9c66d373.arrow
Entailment:  76.93012048192772
Neutral:  80.59397590361446
Contradiction:  80.23614457831326
Samples:  830
--------------------------------------------------
Language: el
WARNING:datasets.builder:Found cached dataset xnli (/root/.cache/huggingface/datasets/xnli/el/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd)
100%
3/3 [00:00<00:00, 45.74it/s]
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/el/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-839eb2a2fba23232.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/el/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-62597a0af99af80f.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/el/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-88535a8382d302a3.arrow
WARNING:datasets.arrow_dataset:Loading cached processed dataset at /root/.cache/huggingface/datasets/xnli/el/1.1.0/818164464f9c9fd15776ca8a00423b074344c3e929d00a2c1a84aa5a50c928bd/cache-18a6b23de04fdc0c.arrow
Entailment:  90.89518072289157
Neutral:  95.09879518072289
Contradiction:  93.93132530120482
Samples:  830
--------------------------------------------------
"""
