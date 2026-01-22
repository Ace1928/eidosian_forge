import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def matchPreviousLiteral(expr):
    """Helper to define an expression that is indirectly defined from
       the tokens matched in a previous expression, that is, it looks
       for a 'repeat' of a previous expression.  For example::
           first = Word(nums)
           second = matchPreviousLiteral(first)
           matchExpr = first + ":" + second
       will match C{"1:1"}, but not C{"1:2"}.  Because this matches a
       previous literal, will also match the leading C{"1:1"} in C{"1:10"}.
       If this is not desired, use C{matchPreviousExpr}.
       Do *not* use with packrat parsing enabled.
    """
    rep = Forward()

    def copyTokenToRepeater(s, l, t):
        if t:
            if len(t) == 1:
                rep << t[0]
            else:
                tflat = _flatten(t.asList())
                rep << And([Literal(tt) for tt in tflat])
        else:
            rep << Empty()
    expr.addParseAction(copyTokenToRepeater, callDuringTry=True)
    return rep