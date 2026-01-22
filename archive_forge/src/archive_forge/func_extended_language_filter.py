from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def extended_language_filter(self, lang_range: str, lang_tag: str) -> bool:
    """Filter the language tags."""
    match = True
    lang_range = RE_WILD_STRIP.sub('-', lang_range).lower()
    ranges = lang_range.split('-')
    subtags = lang_tag.lower().split('-')
    length = len(ranges)
    slength = len(subtags)
    rindex = 0
    sindex = 0
    r = ranges[rindex]
    s = subtags[sindex]
    if length == 1 and slength == 1 and (not r) and (r == s):
        return True
    if r != '*' and r != s or (r == '*' and slength == 1 and (not s)):
        match = False
    rindex += 1
    sindex += 1
    while match and rindex < length:
        r = ranges[rindex]
        try:
            s = subtags[sindex]
        except IndexError:
            match = False
            continue
        if not r:
            match = False
            continue
        elif s == r:
            rindex += 1
        elif len(s) == 1:
            match = False
            continue
        sindex += 1
    return match