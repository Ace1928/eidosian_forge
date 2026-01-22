import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
def bfs_iter(self, include_self=False, right_to_left=False):
    """Breadth first iteration (non-recursive) over the child nodes."""
    return _BFSIter(self, include_self=include_self, right_to_left=right_to_left)