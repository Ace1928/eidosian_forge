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
def root_hypernyms(self):
    """Get the topmost hypernyms of this synset in WordNet."""
    result = []
    seen = set()
    todo = [self]
    while todo:
        next_synset = todo.pop()
        if next_synset not in seen:
            seen.add(next_synset)
            next_hypernyms = next_synset.hypernyms() + next_synset.instance_hypernyms()
            if not next_hypernyms:
                result.append(next_synset)
            else:
                todo.extend(next_hypernyms)
    return result