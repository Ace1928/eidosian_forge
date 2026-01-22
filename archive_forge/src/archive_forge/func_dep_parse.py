import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def dep_parse(self, sentence):
    """
        Return a dependency graph for the sentence.

        :param sentence: the sentence to be parsed
        :type sentence: list(str)
        :rtype: DependencyGraph
        """
    if self.depparser is None:
        from nltk.parse import MaltParser
        self.depparser = MaltParser(tagger=self.get_pos_tagger())
    if not self.depparser._trained:
        self.train_depparser()
    return self.depparser.parse(sentence, verbose=self.verbose)