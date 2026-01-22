import re
import sys
from typing import List, Optional, Union
def maybe_extract_lines(self) -> Optional[List[bytearray]]:
    """
        Extract everything up to the first blank line, and return a list of lines.
        """
    if self._data[:1] == b'\n':
        self._extract(1)
        return []
    if self._data[:2] == b'\r\n':
        self._extract(2)
        return []
    match = blank_line_regex.search(self._data, self._multiple_lines_search)
    if match is None:
        self._multiple_lines_search = max(0, len(self._data) - 2)
        return None
    idx = match.span(0)[-1]
    out = self._extract(idx)
    lines = out.split(b'\n')
    for line in lines:
        if line.endswith(b'\r'):
            del line[-1]
    assert lines[-2] == lines[-1] == b''
    del lines[-2:]
    return lines