from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
def fill_paragraphs(s: str, width: int, sep: str='\n') -> str:
    """Fill paragraphs with newlines (or custom separator)."""
    return sep.join((fill(p, width) for p in s.split(sep)))