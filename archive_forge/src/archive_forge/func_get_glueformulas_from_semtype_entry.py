import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def get_glueformulas_from_semtype_entry(self, lookup, word, node, depgraph, counter):
    glueformulas = []
    glueFormulaFactory = self.get_GlueFormula_factory()
    for meaning, glue in lookup:
        gf = glueFormulaFactory(self.get_meaning_formula(meaning, word), glue)
        if not len(glueformulas):
            gf.word = word
        else:
            gf.word = f'{word}{len(glueformulas) + 1}'
        gf.glue = self.initialize_labels(gf.glue, node, depgraph, counter.get())
        glueformulas.append(gf)
    return glueformulas