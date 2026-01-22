import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def make_imp_expression(first, second):
    if isinstance(first, DRS):
        return DRS(first.refs, first.conds, second)
    if isinstance(first, DrtConcatenation):
        return DrtConcatenation(first.first, first.second, second)
    raise Exception('Antecedent of implication must be a DRS')