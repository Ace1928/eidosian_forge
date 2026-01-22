from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_subselectors(self, el: bs4.Tag, selectors: tuple[ct.SelectorList, ...]) -> bool:
    """Match selectors."""
    match = True
    for sel in selectors:
        if not self.match_selectors(el, sel):
            match = False
    return match