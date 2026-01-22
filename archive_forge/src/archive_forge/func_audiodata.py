import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def audiodata(self, utterance, start=0, end=None):
    assert end is None or end > start
    headersize = 44
    with self.open(utterance + '.wav') as fp:
        if end is None:
            data = fp.read()
        else:
            data = fp.read(headersize + end * 2)
    return data[headersize + start * 2:]