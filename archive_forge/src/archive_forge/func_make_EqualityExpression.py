import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
def make_EqualityExpression(self, first, second):
    return DrtEqualityExpression(first, second)