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
def propagate_semtypes(self):
    """
        Apply inference rules to distribute semtypes over relations between FEs.
        For FrameNet 1.5, this results in 1011 semtypes being propagated.
        (Not done by default because it requires loading all frame files,
        which takes several seconds. If this needed to be fast, it could be rewritten
        to traverse the neighboring relations on demand for each FE semtype.)

        >>> from nltk.corpus import framenet as fn
        >>> x = sum(1 for f in fn.frames() for fe in f.FE.values() if fe.semType)
        >>> fn.propagate_semtypes()
        >>> y = sum(1 for f in fn.frames() for fe in f.FE.values() if fe.semType)
        >>> y-x > 1000
        True
        """
    if not self._semtypes:
        self._loadsemtypes()
    if not self._ferel_idx:
        self._buildrelationindex()
    changed = True
    i = 0
    nPropagations = 0
    while changed:
        i += 1
        changed = False
        for ferel in self.fe_relations():
            superST = ferel.superFE.semType
            subST = ferel.subFE.semType
            try:
                if superST and superST is not subST:
                    assert subST is None or self.semtype_inherits(subST, superST), (superST.name, ferel, subST.name)
                    if subST is None:
                        ferel.subFE.semType = subST = superST
                        changed = True
                        nPropagations += 1
                if ferel.type.name in ['Perspective_on', 'Subframe', 'Precedes'] and subST and (subST is not superST):
                    assert superST is None, (superST.name, ferel, subST.name)
                    ferel.superFE.semType = superST = subST
                    changed = True
                    nPropagations += 1
            except AssertionError as ex:
                continue