import re
import sys
from typing import List, Optional, Union
def maybe_extract_next_line(self) -> Optional[bytearray]:
    """
        Extract the first line, if it is completed in the buffer.
        """
    search_start_index = max(0, self._next_line_search - 1)
    partial_idx = self._data.find(b'\r\n', search_start_index)
    if partial_idx == -1:
        self._next_line_search = len(self._data)
        return None
    idx = partial_idx + 2
    return self._extract(idx)