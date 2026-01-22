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
def ft_sents(self, docNamePattern=None):
    """
        Full-text annotation sentences, optionally filtered by document name.
        """
    return PrettyLazyIteratorList((sent for d in self.docs(docNamePattern) for sent in d.sentence))