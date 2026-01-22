import fnmatch
import os
import string
import sys
from typing import List, Sequence, Iterable, Optional
from .errors import InvalidPathError
def get_subpaths(path: str) -> List[str]:
    """Walk a path and return a list of subpaths."""
    if os.path.isfile(path):
        path = os.path.dirname(path)
    paths = [path]
    path, tail = os.path.split(path)
    while tail:
        paths.append(path)
        path, tail = os.path.split(path)
    return paths