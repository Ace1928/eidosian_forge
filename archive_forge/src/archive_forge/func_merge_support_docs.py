import json
import math
import re
import signal
from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin
def merge_support_docs(doc_name):
    docs = []
    for f_name in glob(pjoin(doc_name, '*.json')):
        docs += json.load(open(f_name))
    print('files loaded, merging')
    merged = {}
    for eli_k, num, article in docs:
        merged[eli_k] = merged.get(eli_k, [''] * 100)
        merged[eli_k][num] = article
    print('articles merged, deduping')
    for eli_k, articles in merged.items():
        merged[eli_k] = [art for art in articles if art != '']
        merged[eli_k] = [x for i, x in enumerate(merged[eli_k]) if x['url'] not in [y['url'] for y in merged[eli_k][:i]]]
    return list(merged.items())