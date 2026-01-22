import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
def read_blankline_block(stream):
    s = ''
    while True:
        line = stream.readline()
        if not line:
            if s:
                return [s]
            else:
                return []
        elif line and (not line.strip()):
            if s:
                return [s]
        else:
            s += line