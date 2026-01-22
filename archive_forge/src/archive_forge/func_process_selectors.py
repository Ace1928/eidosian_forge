from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def process_selectors(self, index: int=0, flags: int=0) -> ct.SelectorList:
    """Process selectors."""
    return self.parse_selectors(self.selector_iter(self.pattern), index, flags)