import bisect
import re
from typing import Dict, List, Tuple
def offset_to_line(self, offset):
    """
    Converts 0-based character offset to pair (line, col) of 1-based line and 0-based column
    numbers.
    """
    offset = max(0, min(self._text_len, offset))
    line_index = bisect.bisect_right(self._line_offsets, offset) - 1
    return (line_index + 1, offset - self._line_offsets[line_index])