from __future__ import annotations
import logging
import re
from ..token import Token
from .state_core import StateCore
def replace_scoped(inlineTokens: list[Token]) -> None:
    inside_autolink = 0
    for token in inlineTokens:
        if token.type == 'text' and (not inside_autolink):
            token.content = SCOPED_ABBR_RE.sub(replaceFn, token.content)
        if token.type == 'link_open' and token.info == 'auto':
            inside_autolink -= 1
        if token.type == 'link_close' and token.info == 'auto':
            inside_autolink += 1