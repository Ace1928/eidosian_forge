import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def to_queue(lz):
    """Parse a pytree of lazy arrays into a queue of nodes, sorted by depth.
    This is useful for traversing the computational graph of multiple outputs
    in topological order.
    """
    if isinstance(lz, LazyArray):
        return [lz]
    queue = tree_flatten(lz, is_lazy_array)
    queue.sort(key=get_depth)
    return queue