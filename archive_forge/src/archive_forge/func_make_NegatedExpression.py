import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def make_NegatedExpression(self, expression):
    return DrtNegatedExpression(expression)