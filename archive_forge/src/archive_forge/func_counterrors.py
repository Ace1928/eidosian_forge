from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template
def counterrors(xs):
    return sum((t[1] != g[1] for pair in zip(xs, gold) for t, g in zip(*pair)))