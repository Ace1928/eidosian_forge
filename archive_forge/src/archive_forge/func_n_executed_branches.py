from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
@property
def n_executed_branches(self) -> int:
    """Returns the number of executed branches."""
    return self.n_branches - self.n_missing_branches