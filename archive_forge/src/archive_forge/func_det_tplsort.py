from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def det_tplsort(tpl_value):
    return (tpl_value[1], repr(tpl_value[0]))