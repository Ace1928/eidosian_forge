import bisect
import re
from typing import Dict, List, Tuple
def line_to_offset(self, line, column):
    """
    Converts 1-based line number and 0-based column to 0-based character offset into text.
    """
    line -= 1
    if line >= len(self._line_offsets):
        return self._text_len
    elif line < 0:
        return 0
    else:
        return min(self._line_offsets[line] + max(0, column), self._text_len)