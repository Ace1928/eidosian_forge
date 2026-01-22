import fnmatch
import os
import string
import sys
from typing import List, Sequence, Iterable, Optional
from .errors import InvalidPathError
def match_item(item: str, filter_pattern: Sequence[str]) -> bool:
    """Check if a string matches one or more fnmatch patterns."""
    if isinstance(filter_pattern, str):
        filter_pattern = [filter_pattern]
    idx = 0
    found = False
    while idx < len(filter_pattern) and (not found):
        found |= fnmatch.fnmatch(item.lower(), filter_pattern[idx].lower())
        idx += 1
    return found