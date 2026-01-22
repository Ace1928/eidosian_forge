import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def initialize_labels(self, expr, node, depgraph, unique_index):
    if isinstance(expr, linearlogic.AtomicExpression):
        name = self.find_label_name(expr.name.lower(), node, depgraph, unique_index)
        if name[0].isupper():
            return linearlogic.VariableExpression(name)
        else:
            return linearlogic.ConstantExpression(name)
    else:
        return linearlogic.ImpExpression(self.initialize_labels(expr.antecedent, node, depgraph, unique_index), self.initialize_labels(expr.consequent, node, depgraph, unique_index))