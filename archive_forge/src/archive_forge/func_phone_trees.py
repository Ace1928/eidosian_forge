import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
def phone_trees(self, utterances=None):
    if utterances is None:
        utterances = self._utterances
    if isinstance(utterances, str):
        utterances = [utterances]
    trees = []
    for utterance in utterances:
        word_times = self.word_times(utterance)
        phone_times = self.phone_times(utterance)
        sent_times = self.sent_times(utterance)
        while sent_times:
            sent, sent_start, sent_end = sent_times.pop(0)
            trees.append(Tree('S', []))
            while word_times and phone_times and (phone_times[0][2] <= word_times[0][1]):
                trees[-1].append(phone_times.pop(0)[0])
            while word_times and word_times[0][2] <= sent_end:
                word, word_start, word_end = word_times.pop(0)
                trees[-1].append(Tree(word, []))
                while phone_times and phone_times[0][2] <= word_end:
                    trees[-1][-1].append(phone_times.pop(0)[0])
            while phone_times and phone_times[0][2] <= sent_end:
                trees[-1].append(phone_times.pop(0)[0])
    return trees