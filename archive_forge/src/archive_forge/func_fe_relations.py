import itertools
import os
import re
import sys
import textwrap
import types
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from operator import itemgetter
from pprint import pprint
from nltk.corpus.reader import XMLCorpusReader, XMLCorpusView
from nltk.util import LazyConcatenation, LazyIteratorList, LazyMap
def fe_relations(self):
    """
        Obtain a list of frame element relations.

        >>> from nltk.corpus import framenet as fn
        >>> ferels = fn.fe_relations()
        >>> isinstance(ferels, list)
        True
        >>> len(ferels) in (10020, 12393)   # FN 1.5 and 1.7, resp.
        True
        >>> PrettyDict(ferels[0], breakLines=True) # doctest: +NORMALIZE_WHITESPACE
        {'ID': 14642,
        '_type': 'ferelation',
        'frameRelation': <Parent=Abounding_with -- Inheritance -> Child=Lively_place>,
        'subFE': <fe ID=11370 name=Degree>,
        'subFEName': 'Degree',
        'subFrame': <frame ID=1904 name=Lively_place>,
        'subID': 11370,
        'supID': 2271,
        'superFE': <fe ID=2271 name=Degree>,
        'superFEName': 'Degree',
        'superFrame': <frame ID=262 name=Abounding_with>,
        'type': <framerelationtype ID=1 name=Inheritance>}

        :return: A list of all of the frame element relations in framenet
        :rtype: list(dict)
        """
    if not self._ferel_idx:
        self._buildrelationindex()
    return PrettyList(sorted(self._ferel_idx.values(), key=lambda ferel: (ferel.type.ID, ferel.frameRelation.superFrameName, ferel.superFEName, ferel.frameRelation.subFrameName, ferel.subFEName)))