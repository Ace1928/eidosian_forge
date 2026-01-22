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
def semtype(self, key):
    """
        >>> from nltk.corpus import framenet as fn
        >>> fn.semtype(233).name
        'Temperature'
        >>> fn.semtype(233).abbrev
        'Temp'
        >>> fn.semtype('Temperature').ID
        233

        :param key: The name, abbreviation, or id number of the semantic type
        :type key: string or int
        :return: Information about a semantic type
        :rtype: dict
        """
    if isinstance(key, int):
        stid = key
    else:
        try:
            stid = self._semtypes[key]
        except TypeError:
            self._loadsemtypes()
            stid = self._semtypes[key]
    try:
        st = self._semtypes[stid]
    except TypeError:
        self._loadsemtypes()
        st = self._semtypes[stid]
    return st