import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def match_previous_literal(expr: ParserElement) -> ParserElement:
    """Helper to define an expression that is indirectly defined from
    the tokens matched in a previous expression, that is, it looks for
    a 'repeat' of a previous expression.  For example::

        first = Word(nums)
        second = match_previous_literal(first)
        match_expr = first + ":" + second

    will match ``"1:1"``, but not ``"1:2"``.  Because this
    matches a previous literal, will also match the leading
    ``"1:1"`` in ``"1:10"``. If this is not desired, use
    :class:`match_previous_expr`. Do *not* use with packrat parsing
    enabled.
    """
    rep = Forward()

    def copy_token_to_repeater(s, l, t):
        if t:
            if len(t) == 1:
                rep << t[0]
            else:
                tflat = _flatten(t.as_list())
                rep << And((Literal(tt) for tt in tflat))
        else:
            rep << Empty()
    expr.add_parse_action(copy_token_to_repeater, callDuringTry=True)
    rep.set_name('(prev) ' + str(expr))
    return rep