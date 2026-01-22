import json
import math
import re
import signal
from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin
def tf_idf_vec(sentence, vocounts, totcounts):
    sen_tab = sentence.lower().split()
    uni_dic = {}
    for _, w in enumerate(sen_tab):
        uni_dic[w] = -math.log(float(vocounts.get(w, 1.0)) / totcounts)
    for i in range(len(sen_tab)):
        a = sen_tab[i]
        if i + 1 < len(sen_tab):
            b = sen_tab[i + 1]
            uni_dic[a + ' ' + b] = uni_dic[a] + uni_dic[b]
        if i + 2 < len(sen_tab):
            b = sen_tab[i + 2]
            uni_dic[a + ' ' + b] = uni_dic[a] + uni_dic[b]
    uni_norm = math.sqrt(sum([x * x for x in uni_dic.values()]))
    if uni_norm > 0:
        for w in uni_dic:
            uni_dic[w] /= uni_norm
    return uni_dic