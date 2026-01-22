import re
from collections import defaultdict
from nltk.ccg.api import CCGVar, Direction, FunctionalCategory, PrimitiveCategory
from nltk.internals import deprecated
from nltk.sem.logic import Expression
def augParseCategory(line, primitives, families, var=None):
    """
    Parse a string representing a category, and returns a tuple with
    (possibly) the CCG variable for the category
    """
    cat_string, rest = nextCategory(line)
    if cat_string.startswith('('):
        res, var = augParseCategory(cat_string[1:-1], primitives, families, var)
    else:
        res, var = parsePrimitiveCategory(PRIM_RE.match(cat_string).groups(), primitives, families, var)
    while rest != '':
        app = APP_RE.match(rest).groups()
        direction = parseApplication(app[0:3])
        rest = app[3]
        cat_string, rest = nextCategory(rest)
        if cat_string.startswith('('):
            arg, var = augParseCategory(cat_string[1:-1], primitives, families, var)
        else:
            arg, var = parsePrimitiveCategory(PRIM_RE.match(cat_string).groups(), primitives, families, var)
        res = FunctionalCategory(res, arg, direction)
    return (res, var)