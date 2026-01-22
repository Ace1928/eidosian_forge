import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def count_chars_in_width(line_str: str, max_width: int) -> int:
    """Count the number of characters in `line_str` that would fit in a
    terminal or editor of `max_width` (which respects Unicode East Asian
    Width).
    """
    total_width = 0
    for i, char in enumerate(line_str):
        width = char_width(char)
        if width + total_width > max_width:
            return i
        total_width += width
    return len(line_str)