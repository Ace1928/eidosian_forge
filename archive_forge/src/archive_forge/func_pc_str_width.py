from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def pc_str_width(self) -> int:
    """How many characters wide can pc_covered_str be?"""
    width = 3
    if self._precision > 0:
        width += 1 + self._precision
    return width