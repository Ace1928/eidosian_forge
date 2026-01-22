from __future__ import annotations
import collections
from typing import Callable, Iterable, TYPE_CHECKING
from coverage.debug import auto_repr
from coverage.exceptions import ConfigError
from coverage.misc import nice_pair
from coverage.types import TArc, TLineNo
def branch_stats(self) -> dict[TLineNo, tuple[int, int]]:
    """Get stats about branches.

        Returns a dict mapping line numbers to a tuple:
        (total_exits, taken_exits).
        """
    missing_arcs = self.missing_branch_arcs()
    stats = {}
    for lnum in self._branch_lines():
        exits = self.exit_counts[lnum]
        missing = len(missing_arcs[lnum])
        stats[lnum] = (exits, exits - missing)
    return stats