import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def matchPreviousExpr(expr):
    """Helper to define an expression that is indirectly defined from
       the tokens matched in a previous expression, that is, it looks
       for a 'repeat' of a previous expression.  For example::
           first = Word(nums)
           second = matchPreviousExpr(first)
           matchExpr = first + ":" + second
       will match C{"1:1"}, but not C{"1:2"}.  Because this matches by
       expressions, will *not* match the leading C{"1:1"} in C{"1:10"};
       the expressions are evaluated first, and then compared, so
       C{"1"} is compared with C{"10"}.
       Do *not* use with packrat parsing enabled.
    """
    rep = Forward()
    e2 = expr.copy()
    rep << e2

    def copyTokenToRepeater(s, l, t):
        matchTokens = _flatten(t.asList())

        def mustMatchTheseTokens(s, l, t):
            theseTokens = _flatten(t.asList())
            if theseTokens != matchTokens:
                raise ParseException('', 0, '')
        rep.setParseAction(mustMatchTheseTokens, callDuringTry=True)
    expr.addParseAction(copyTokenToRepeater, callDuringTry=True)
    return rep