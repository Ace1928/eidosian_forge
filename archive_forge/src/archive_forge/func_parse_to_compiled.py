import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def parse_to_compiled(self, sentence):
    gfls = [self.depgraph_to_glue(dg) for dg in self.dep_parse(sentence)]
    return [self.gfl_to_compiled(gfl) for gfl in gfls]