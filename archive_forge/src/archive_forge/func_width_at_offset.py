import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def width_at_offset(self, n: int) -> int:
    """Returns the horizontal position of character n of the string"""
    width = wcswidth(self.s, n)
    assert width != -1
    return width