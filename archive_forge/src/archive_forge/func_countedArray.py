import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def countedArray(expr, intExpr=None):
    """Helper to define a counted list of expressions.
       This helper defines a pattern of the form::
           integer expr expr expr...
       where the leading integer tells how many expr expressions follow.
       The matched tokens returns the array of expr tokens as a list - the leading count token is suppressed.
    """
    arrayExpr = Forward()

    def countFieldParseAction(s, l, t):
        n = t[0]
        arrayExpr << (n and Group(And([expr] * n)) or Group(empty))
        return []
    if intExpr is None:
        intExpr = Word(nums).setParseAction(lambda t: int(t[0]))
    else:
        intExpr = intExpr.copy()
    intExpr.setName('arrayLen')
    intExpr.addParseAction(countFieldParseAction, callDuringTry=True)
    return intExpr + arrayExpr