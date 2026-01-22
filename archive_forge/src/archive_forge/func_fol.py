import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def fol(self):
    return ApplicationExpression(self.function.fol(), self.argument.fol())