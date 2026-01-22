import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def history_size_footprint(self, include_inputs=True):
    """Get the combined size of intermediates at each step of the
        computation. Note this assumes that intermediates are immediately
        garbage collected when they are no longer required.

        Parameters
        ----------
        include_inputs : bool, optional
            Whether to include the size of the inputs in the computation. If
            ``True`` It is assumed they can be garbage collected once used but
            are all present at the beginning of the computation.
        """
    delete_checked = set()
    sizes = []
    input_size = 0
    for node in reversed(tuple(self.ascend())):
        for c in node._deps:
            if c not in delete_checked:
                if include_inputs or c._deps:
                    sizes.append(-c.size)
                delete_checked.add(c)
        if node._data is None:
            sizes.append(+node.size)
        elif include_inputs:
            input_size += node.size
    sizes.append(input_size)
    sizes.reverse()
    return list(itertools.accumulate(sizes))