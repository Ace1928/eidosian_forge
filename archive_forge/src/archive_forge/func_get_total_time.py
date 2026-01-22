import re
import operator
import heapq
from collections import namedtuple
from collections.abc import Sequence
from contextlib import contextmanager
from functools import cached_property
from numba.core import config
import llvmlite.binding as llvm
def get_total_time(self):
    """Computes the sum of the total time across all contained timings.

        Returns
        -------
        res: float or None
            Returns the total number of seconds or None if no timings were
            recorded
        """
    if self._records:
        return sum((r.timings.get_total_time() for r in self._records))
    else:
        return None