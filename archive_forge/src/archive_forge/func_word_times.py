import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def word_times(self, utterances=None):
    results = []
    for fileid in self._utterance_fileids(utterances, '.wrd'):
        with self.open(fileid) as fp:
            for line in fp:
                if line.strip():
                    results.append((line.split()[2], int(line.split()[0]), int(line.split()[1])))
    return results