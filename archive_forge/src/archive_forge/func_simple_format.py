from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
def simple_format(s: str, keys: dict[str, str | Callable], pattern: Pattern[str]=RE_FORMAT, expand: str='\\1') -> str:
    """Format string, expanding abbreviations in keys'."""
    if s:
        keys.setdefault('%', '%')

        def resolve(match: Match) -> str | Any:
            key = match.expand(expand)
            try:
                resolver = keys[key]
            except KeyError:
                raise ValueError(UNKNOWN_SIMPLE_FORMAT_KEY.format(key, s))
            if callable(resolver):
                return resolver()
            return resolver
        return pattern.sub(resolve, s)
    return s