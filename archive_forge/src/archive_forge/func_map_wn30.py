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
def map_wn30(self):
    """Mapping from Wordnet 3.0 to currently loaded Wordnet version"""
    if self.get_version() == '3.0':
        return None
    else:
        return self.map_to_one()