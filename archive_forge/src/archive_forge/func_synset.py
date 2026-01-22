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
def synset(self, name):
    lemma, pos, synset_index_str = name.lower().rsplit('.', 2)
    synset_index = int(synset_index_str) - 1
    try:
        offset = self._lemma_pos_offset_map[lemma][pos][synset_index]
    except KeyError as e:
        raise WordNetError(f'No lemma {lemma!r} with part of speech {pos!r}') from e
    except IndexError as e:
        n_senses = len(self._lemma_pos_offset_map[lemma][pos])
        raise WordNetError(f'Lemma {lemma!r} with part of speech {pos!r} only has {n_senses} {('sense' if n_senses == 1 else 'senses')}') from e
    synset = self.synset_from_pos_and_offset(pos, offset)
    if pos == 's' and synset._pos == 'a':
        message = 'Adjective satellite requested but only plain adjective found for lemma %r'
        raise WordNetError(message % lemma)
    assert synset._pos == pos or (pos == 'a' and synset._pos == 's')
    return synset