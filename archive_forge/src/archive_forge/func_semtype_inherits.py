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
def semtype_inherits(self, st, superST):
    if not isinstance(st, dict):
        st = self.semtype(st)
    if not isinstance(superST, dict):
        superST = self.semtype(superST)
    par = st.superType
    while par:
        if par is superST:
            return True
        par = par.superType
    return False