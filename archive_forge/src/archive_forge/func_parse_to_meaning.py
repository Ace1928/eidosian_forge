import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def parse_to_meaning(self, sentence):
    readings = []
    for agenda in self.parse_to_compiled(sentence):
        readings.extend(self.get_readings(agenda))
    return readings