from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_pseudo_contains(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
    """Parse contains."""
    pseudo = util.lower(css_unescape(m.group('name')))
    if pseudo == ':contains':
        warnings.warn("The pseudo class ':contains' is deprecated, ':-soup-contains' should be used moving forward.", FutureWarning)
    contains_own = pseudo == ':-soup-contains-own'
    values = css_unescape(m.group('values'))
    patterns = []
    for token in RE_VALUES.finditer(values):
        if token.group('split'):
            continue
        value = token.group('value')
        if value.startswith(("'", '"')):
            value = css_unescape(value[1:-1], True)
        else:
            value = css_unescape(value)
        patterns.append(value)
    sel.contains.append(ct.SelectorContains(patterns, contains_own))
    has_selector = True
    return has_selector