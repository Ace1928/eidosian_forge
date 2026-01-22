import re
from collections import defaultdict
from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression
def nextCategory(string):
    """
    Separate the string for the next portion of the category from the rest
    of the string
    """
    if string.startswith('('):
        return matchBrackets(string)
    return NEXTPRIM_RE.match(string).groups()