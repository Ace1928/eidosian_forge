import re
import sys
from functools import lru_cache
from typing import Final, List, Match, Pattern
from black._width_table import WIDTH_TABLE
from blib2to3.pytree import Leaf
def str_width(line_str: str) -> int:
    """Return the width of `line_str` as it would be displayed in a terminal
    or editor (which respects Unicode East Asian Width).

    You could utilize this function to determine, for example, if a string
    is too wide to display in a terminal or editor.
    """
    if line_str.isascii():
        return len(line_str)
    return sum(map(char_width, line_str))