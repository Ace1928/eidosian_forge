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
def lemma_names(self, lang='eng'):
    """Return all the lemma_names associated with the synset"""
    if lang == 'eng':
        return self._lemma_names
    else:
        reader = self._wordnet_corpus_reader
        reader._load_lang_data(lang)
        i = reader.ss2of(self)
        if i in reader._lang_data[lang][0]:
            return reader._lang_data[lang][0][i]
        else:
            return []