from __future__ import annotations
from fnmatch import fnmatch
from re import match as rematch
from typing import Callable, cast
from .utils.compat import entrypoints
from .utils.encoding import bytes_to_str
def register_glob() -> None:
    """Register glob into default registry."""
    registry.register('glob', fnmatch)