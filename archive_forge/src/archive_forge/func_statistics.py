from collections.abc import Sequence, Iterable
from functools import total_ordering
import fnmatch
import linecache
import os.path
import pickle
from _tracemalloc import *
from _tracemalloc import _get_object_traceback, _get_traces
def statistics(self, key_type, cumulative=False):
    """
        Group statistics by key_type. Return a sorted list of Statistic
        instances.
        """
    grouped = self._group_by(key_type, cumulative)
    statistics = list(grouped.values())
    statistics.sort(reverse=True, key=Statistic._sort_key)
    return statistics