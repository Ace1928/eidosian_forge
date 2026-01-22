from __future__ import annotations
import re
from functools import lru_cache
from . import util
from . import css_match as cm
from . import css_types as ct
from .util import SelectorSyntaxError
import warnings
from typing import Match, Any, Iterator, cast
def parse_tag_pattern(self, sel: _Selector, m: Match[str], has_selector: bool) -> bool:
    """Parse tag pattern from regex match."""
    prefix = css_unescape(m.group('tag_ns')[:-1]) if m.group('tag_ns') else None
    tag = css_unescape(m.group('tag_name'))
    sel.tag = ct.SelectorTag(tag, prefix)
    has_selector = True
    return has_selector