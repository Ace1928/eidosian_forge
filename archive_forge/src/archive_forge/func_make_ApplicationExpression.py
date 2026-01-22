import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def make_ApplicationExpression(self, function, argument):
    return DrtApplicationExpression(function, argument)