import operator
import re
from collections import defaultdict
from functools import reduce, total_ordering
from nltk.internals import Counter
from nltk.util import Trie
def skolem_function(univ_scope=None):
    """
    Return a skolem function over the variables in univ_scope
    param univ_scope
    """
    skolem = VariableExpression(Variable('F%s' % _counter.get()))
    if univ_scope:
        for v in list(univ_scope):
            skolem = skolem(VariableExpression(v))
    return skolem