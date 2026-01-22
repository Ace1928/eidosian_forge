import re
import sys
from typing import List, Optional, Union
def maybe_extract_at_most(self, count: int) -> Optional[bytearray]:
    """
        Extract a fixed number of bytes from the buffer.
        """
    out = self._data[:count]
    if not out:
        return None
    return self._extract(count)