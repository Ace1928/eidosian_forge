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
def lemma_from_key(self, key):
    key = key.lower()
    lemma_name, lex_sense = key.split('%')
    pos_number, lexname_index, lex_id, _, _ = lex_sense.split(':')
    pos = self._pos_names[int(pos_number)]
    if self._key_synset_file is None:
        self._key_synset_file = self.open('index.sense')
    synset_line = _binary_search_file(self._key_synset_file, key)
    if not synset_line:
        raise WordNetError('No synset found for key %r' % key)
    offset = int(synset_line.split()[1])
    synset = self.synset_from_pos_and_offset(pos, offset)
    for lemma in synset._lemmas:
        if lemma._key == key:
            return lemma
    raise WordNetError('No lemma found for for key %r' % key)