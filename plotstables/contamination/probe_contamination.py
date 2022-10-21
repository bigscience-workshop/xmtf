import nltk 
import cld3
from multiprocessing import Pool
import threading
from concurrent.futures import ThreadPoolExecutor

nltk.download("punkt", quiet=True)
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

splitter = nltk.load("tokenizers/punkt/english.pickle")
nltk_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())

from tqdm import tqdm 
import json
import argparse

parser = argparse.ArgumentParser(description='Chunk')
parser.add_argument('--start_idx', type=int)
args = parser.parse_args()
lines_total = 50913583
lines_threads = 64
lines_interval = lines_total // lines_threads

lines_start_idx = args.start_idx * lines_interval
lines_end_idx = min(lines_total, (args.start_idx+1) * lines_interval)

bloom_dir = "/tmp/"

file_name = "roots_1e-1_train"
out_file = "roots_1e-1_meta_mp"
import itertools
import os

def process_lines(line):
    data = json.loads(line)
    meta_data = dict( data )
    meta_data.pop('text')
    text = data['text']
    meta_data['cld3_language'] = []
    meta_data['cld3_confidence'] = []
    meta_data['cld3_reliable'] = []
    for sentence in nltk_splitter.tokenize(text):
        detect_res = cld3.get_language(sentence)
        meta_data['cld3_language'].append( detect_res.language )
        meta_data['cld3_confidence'].append( detect_res.probability )
        meta_data['cld3_reliable'].append( detect_res.is_reliable )
    return meta_data

line_idx = 0
line_idxs = list(range(lines_start_idx, lines_end_idx))
print(lines_start_idx, lines_end_idx)
with open(f'{bloom_dir}/{file_name}.jsonl', 'r', encoding='utf-8') as fin \
, open(f'{bloom_dir}/mp/{out_file}_{args.start_idx}.jsonl', "w") as fout:
    lines = itertools.islice(fin, lines_start_idx, lines_end_idx)
    for line in tqdm(lines):
        meta_data = process_lines(line)
        fout.write(f"{json.dumps(meta_data)}\n")
    