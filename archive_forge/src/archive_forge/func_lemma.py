import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
def lemma(self, name, lang='eng'):
    """Return lemma object that matches the name"""
    separator = SENSENUM_RE.search(name).end()
    synset_name, lemma_name = (name[:separator - 1], name[separator:])
    synset = self.synset(synset_name)
    for lemma in synset.lemmas(lang):
        if lemma._name == lemma_name:
            return lemma
    raise WordNetError(f'No lemma {lemma_name!r} in {synset_name!r}')