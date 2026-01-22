import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def copy_token_to_repeater(s, l, t):
    matchTokens = _flatten(t.as_list())

    def must_match_these_tokens(s, l, t):
        theseTokens = _flatten(t.as_list())
        if theseTokens != matchTokens:
            raise ParseException(s, l, f'Expected {matchTokens}, found{theseTokens}')
    rep.set_parse_action(must_match_these_tokens, callDuringTry=True)