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
def read_wordpunct_block(stream):
    toks = []
    for i in range(20):
        toks.extend(wordpunct_tokenize(stream.readline()))
    return toks