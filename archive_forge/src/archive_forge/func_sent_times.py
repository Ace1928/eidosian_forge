import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def sent_times(self, utterances=None):
    return [(line.split(None, 2)[-1].strip(), int(line.split()[0]), int(line.split()[1])) for fileid in self._utterance_fileids(utterances, '.txt') for line in self.open(fileid) if line.strip()]