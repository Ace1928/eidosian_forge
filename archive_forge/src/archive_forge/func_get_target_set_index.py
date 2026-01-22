from __future__ import annotations
import collections.abc as c
import os
import typing as t
from .....io import (
from .....util import (
from .. import (
def get_target_set_index(data: set[int], target_set_indexes: TargetSetIndexes) -> int:
    """Find or add the target set in the result set and return the target set index."""
    return target_set_indexes.setdefault(frozenset(data), len(target_set_indexes))