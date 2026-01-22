import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def match_previous_expr(expr: ParserElement) -> ParserElement:
    """Helper to define an expression that is indirectly defined from
    the tokens matched in a previous expression, that is, it looks for
    a 'repeat' of a previous expression.  For example::

        first = Word(nums)
        second = match_previous_expr(first)
        match_expr = first + ":" + second

    will match ``"1:1"``, but not ``"1:2"``.  Because this
    matches by expressions, will *not* match the leading ``"1:1"``
    in ``"1:10"``; the expressions are evaluated first, and then
    compared, so ``"1"`` is compared with ``"10"``. Do *not* use
    with packrat parsing enabled.
    """
    rep = Forward()
    e2 = expr.copy()
    rep <<= e2

    def copy_token_to_repeater(s, l, t):
        matchTokens = _flatten(t.as_list())

        def must_match_these_tokens(s, l, t):
            theseTokens = _flatten(t.as_list())
            if theseTokens != matchTokens:
                raise ParseException(s, l, f'Expected {matchTokens}, found{theseTokens}')
        rep.set_parse_action(must_match_these_tokens, callDuringTry=True)
    expr.add_parse_action(copy_token_to_repeater, callDuringTry=True)
    rep.set_name('(prev) ' + str(expr))
    return rep