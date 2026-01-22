from __future__ import annotations
import hashlib
import ntpath
import os
import os.path
import posixpath
import re
import sys
from typing import Callable, Iterable
from coverage import env
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted, isolate_module, join_regex
def source_exists(path: str) -> bool:
    """Determine if a source file path exists."""
    if os.path.exists(path):
        return True
    if zip_location(path):
        return True
    return False