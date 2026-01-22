import os
from pathlib import Path
import stat
from itertools import islice, chain
from typing import Iterable, Optional, List, TextIO
from .translations import _
from .filelock import FileLock
def load_from(self, fd: TextIO) -> List[str]:
    entries: List[str] = []
    for line in fd:
        self.append_to(entries, line)
    return entries if len(entries) else ['']