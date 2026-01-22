from __future__ import annotations
import itertools
import re
from enum import Enum
from typing import Hashable, TypeVar
from prompt_toolkit.cache import SimpleCache
from .base import (
from .named_colors import NAMED_COLORS
def merge_styles(styles: list[BaseStyle]) -> _MergedStyle:
    """
    Merge multiple `Style` objects.
    """
    styles = [s for s in styles if s is not None]
    return _MergedStyle(styles)