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
def map_to_many(self):
    sensekey_map1 = self.index_sense('wordnet')
    sensekey_map2 = self.index_sense()
    synset_to_many = {}
    for synsetid in set(sensekey_map1.values()):
        synset_to_many[synsetid] = []
    for sensekey in set(sensekey_map1.keys()).intersection(set(sensekey_map2.keys())):
        source = sensekey_map1[sensekey]
        target = sensekey_map2[sensekey]
        synset_to_many[source].append(target)
    return synset_to_many