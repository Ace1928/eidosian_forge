from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_pseudo_open(self, sel: _Selector, name: str, has_selector: bool, iselector: Iterator[tuple[str, Match[str]]], index: int) -> bool:
    """Parse pseudo with opening bracket."""
    flags = FLG_PSEUDO | FLG_OPEN
    if name == ':not':
        flags |= FLG_NOT
    elif name == ':has':
        flags |= FLG_RELATIVE
    elif name in (':where', ':is'):
        flags |= FLG_FORGIVE
    sel.selectors.append(self.parse_selectors(iselector, index, flags))
    has_selector = True
    return has_selector