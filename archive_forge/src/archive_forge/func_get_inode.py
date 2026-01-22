from __future__ import annotations
import errno
import os
from stat import S_ISDIR
from typing import Any, Callable, Iterator, List, Optional, Tuple
def get_inode(directory: DirectorySnapshot, full_path: str) -> int | Tuple[int, int]:
    return directory.inode(full_path)