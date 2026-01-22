from __future__ import annotations
import re
import sys
import warnings
from functools import wraps, lru_cache
from itertools import count
from typing import TYPE_CHECKING, Generic, Iterator, NamedTuple, TypeVar, TypedDict, overload
def parseBoolValue(value: str | None, fail_on_errors: bool=True, preserve_none: bool=False) -> bool | None:
    """Parses a string representing a boolean value. If parsing was successful,
       returns `True` or `False`. If `preserve_none=True`, returns `True`, `False`,
       or `None`. If parsing was not successful, raises `ValueError`, or, if
       `fail_on_errors=False`, returns `None`."""
    if not isinstance(value, str):
        if preserve_none and value is None:
            return value
        return bool(value)
    elif preserve_none and value.lower() == 'none':
        return None
    elif value.lower() in ('true', 'yes', 'y', 'on', '1'):
        return True
    elif value.lower() in ('false', 'no', 'n', 'off', '0', 'none'):
        return False
    elif fail_on_errors:
        raise ValueError('Cannot parse bool value: %r' % value)