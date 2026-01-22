from __future__ import annotations
import collections.abc as c
import re
import typing as t
from .....executor import (
from .....provisioning import (
from . import (
from . import (
def path_filter_func(path: str) -> bool:
    """Return True if the given path should be included, otherwise return False."""
    if include_path and (not re.search(include_path, path)):
        return False
    if exclude_path and re.search(exclude_path, path):
        return False
    return True