from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
@property
def pc_covered(self) -> float:
    """Returns a single percentage value for coverage."""
    if self.n_statements > 0:
        numerator, denominator = self.ratio_covered
        pc_cov = 100.0 * numerator / denominator
    else:
        pc_cov = 100.0
    return pc_cov