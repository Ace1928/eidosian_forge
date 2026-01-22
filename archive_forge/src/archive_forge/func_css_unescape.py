from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def css_unescape(content: str, string: bool=False) -> str:
    """
    Unescape CSS value.

    Strings allow for spanning the value on multiple strings by escaping a new line.
    """

    def replace(m: Match[str]) -> str:
        """Replace with the appropriate substitute."""
        if m.group(1):
            codepoint = int(m.group(1)[1:], 16)
            if codepoint == 0:
                codepoint = UNICODE_REPLACEMENT_CHAR
            value = chr(codepoint)
        elif m.group(2):
            value = m.group(2)[1:]
        elif m.group(3):
            value = '�'
        else:
            value = ''
        return value
    return (RE_CSS_ESC if not string else RE_CSS_STR_ESC).sub(replace, content)