import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def equality_preds():
    """
    Equality predicates
    """
    names = ['equality', 'inequality']
    for pair in zip(names, [Tokens.EQ, Tokens.NEQ]):
        print('%-15s\t%s' % pair)