from __future__ import annotations
import io
import re
from functools import partial
from pprint import pformat
from re import Match
from textwrap import fill
from typing import Any, Callable, Pattern
def match_case(s: str, other: str) -> str:
    return s.upper() if other.isupper() else s.lower()