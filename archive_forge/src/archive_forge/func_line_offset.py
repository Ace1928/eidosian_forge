from __future__ import annotations
import re
import importlib.util
import sys
from typing import TYPE_CHECKING, Sequence
@property
def line_offset(self) -> int:
    """Returns char index in `self.rawdata` for the start of the current line. """
    for ii in range(len(self.lineno_start_cache) - 1, self.lineno - 1):
        last_line_start_pos = self.lineno_start_cache[ii]
        lf_pos = self.rawdata.find('\n', last_line_start_pos)
        if lf_pos == -1:
            lf_pos = len(self.rawdata)
        self.lineno_start_cache.append(lf_pos + 1)
    return self.lineno_start_cache[self.lineno - 1]