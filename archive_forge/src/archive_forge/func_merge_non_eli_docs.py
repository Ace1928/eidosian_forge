import os
import json
from parlai.core.params import ParlaiParser
from os.path import join as pjoin
from os.path import isdir, isfile
from glob import glob
from data_utils import merge_support_docs
def merge_non_eli_docs(doc_name):
    docs = []
    merged = {}
    for f_name in glob(pjoin(doc_name, '*.json')):
        docs += json.load(open(f_name))
    if not docs or len(docs[0]) < 3:
        for i, (num, article) in enumerate(docs):
            merged[i] = merged.get(i, [''] * 100)
            merged[i][num] = article
    else:
        return None
    for eli_k, articles in merged.items():
        merged[eli_k] = [art for art in articles if art != '']
        merged[eli_k] = [x for i, x in enumerate(merged[eli_k]) if x['url'] not in [y['url'] for y in merged[eli_k][:i]]]
    return list(merged.items())