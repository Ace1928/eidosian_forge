from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
def ensure_sep(sep: str, s: str, n: int=2) -> str:
    """Ensure text s ends in separator sep'."""
    return s + sep * (n - s.count(sep))