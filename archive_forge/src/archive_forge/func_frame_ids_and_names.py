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
def frame_ids_and_names(self, name=None):
    """
        Uses the frame index, which is much faster than looking up each frame definition
        if only the names and IDs are needed.
        """
    if not self._frame_idx:
        self._buildframeindex()
    return {fID: finfo.name for fID, finfo in self._frame_idx.items() if name is None or re.search(name, finfo.name) is not None}