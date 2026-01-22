import json
import math
import re
import signal
from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin
def pre_word_url_tokenize(stp):
    url_list = list(set(re.findall(URL_REGEX, stp)))
    for i, url in enumerate(url_list):
        stp = stp.replace(url, ' URL_%d ' % (i,))
    for a, b in html_pairs:
        stp = stp.replace(a, b)
    pre_txt = ' '.join([str(x) for x in tokenizer(stp)])
    return (' '.join(pre_txt.split()), url_list)