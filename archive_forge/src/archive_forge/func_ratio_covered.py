from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
@property
def ratio_covered(self) -> tuple[int, int]:
    """Return a numerator and denominator for the coverage ratio."""
    numerator = self.n_executed + self.n_executed_branches
    denominator = self.n_statements + self.n_branches
    return (numerator, denominator)