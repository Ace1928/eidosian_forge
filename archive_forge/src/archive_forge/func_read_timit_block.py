import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def read_timit_block(stream):
    """
    Block reader for timit tagged sentences, which are preceded by a sentence
    number that will be ignored.
    """
    line = stream.readline()
    if not line:
        return []
    n, sent = line.split(' ', 1)
    return [sent]