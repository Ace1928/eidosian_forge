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
def custom_lemmas(self, tab_file, lang):
    """
        Reads a custom tab file containing mappings of lemmas in the given
        language to Princeton WordNet 3.0 synset offsets, allowing NLTK's
        WordNet functions to then be used with that language.

        See the "Tab files" section at https://omwn.org/omw1.html for
        documentation on the Multilingual WordNet tab file format.

        :param tab_file: Tab file as a file or file-like object
        :type: lang str
        :param: lang ISO 639-3 code of the language of the tab file
        """
    lg = lang.split('_')[0]
    if len(lg) != 3:
        raise ValueError('lang should be a (3 character) ISO 639-3 code')
    self._lang_data[lang] = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
    for line in tab_file.readlines():
        if isinstance(line, bytes):
            line = line.decode('utf-8')
        if not line.startswith('#'):
            triple = line.strip().split('\t')
            if len(triple) < 3:
                continue
            offset_pos, label = triple[:2]
            val = triple[-1]
            if self.map30:
                if offset_pos in self.map30:
                    offset_pos = self.map30[offset_pos]
                else:
                    if offset_pos not in self.nomap and offset_pos.replace('a', 's') not in self.nomap:
                        warnings.warn(f"{lang}: invalid offset {offset_pos} in '{line}'")
                    continue
            elif offset_pos[-1] == 'a':
                wnss = self.of2ss(offset_pos)
                if wnss and wnss.pos() == 's':
                    offset_pos = self.ss2of(wnss)
            pair = label.split(':')
            attr = pair[-1]
            if len(pair) == 1 or pair[0] == lg:
                if attr == 'lemma':
                    val = val.strip().replace(' ', '_')
                    self._lang_data[lang][1][val.lower()].append(offset_pos)
                if attr in self.lg_attrs:
                    self._lang_data[lang][self.lg_attrs.index(attr)][offset_pos].append(val)