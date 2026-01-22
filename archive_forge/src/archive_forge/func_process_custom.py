from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def process_custom(custom: ct.CustomSelectors | None) -> dict[str, str | ct.SelectorList]:
    """Process custom."""
    custom_selectors = {}
    if custom is not None:
        for key, value in custom.items():
            name = util.lower(key)
            if RE_CUSTOM.match(name) is None:
                raise SelectorSyntaxError(f"The name '{name}' is not a valid custom pseudo-class name")
            if name in custom_selectors:
                raise KeyError(f"The custom selector '{name}' has already been registered")
            custom_selectors[css_unescape(name)] = value
    return custom_selectors