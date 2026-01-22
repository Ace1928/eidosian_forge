import operator
from collections import defaultdict
from functools import reduce
from nltk.inference.api import BaseProverCommand, Prover
from nltk.sem import skolemize
from nltk.sem.logic import (
def clausify(expression):
    """
    Skolemize, clausify, and standardize the variables apart.
    """
    clause_list = []
    for clause in _clausify(skolemize(expression)):
        for free in clause.free():
            if is_indvar(free.name):
                newvar = VariableExpression(unique_variable())
                clause = clause.replace(free, newvar)
        clause_list.append(clause)
    return clause_list