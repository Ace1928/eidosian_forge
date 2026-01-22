import os
from pathlib import Path
import stat
from itertools import islice, chain
from typing import Iterable, Optional, List, TextIO
from .translations import _
from .filelock import FileLock
def find_match_forward(self, search_term: str, include_current: bool=False) -> int:
    add = 0 if include_current else 1
    end = max(0, self.index - (1 - add))
    for idx in range(end):
        val = self.entries_by_index[end - 1 - idx]
        if val.startswith(search_term):
            return idx + (0 if include_current else 1)
    return self.index