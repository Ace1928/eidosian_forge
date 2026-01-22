import os
from pathlib import Path
import stat
from itertools import islice, chain
from typing import Iterable, Optional, List, TextIO
from .translations import _
from .filelock import FileLock
@property
def entries_by_index(self) -> List[str]:
    return list(chain((self.saved_line,), reversed(self.entries)))