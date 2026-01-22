import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
def path_iter(self, include_self=True):
    """Yields back the path from this node to the root node."""
    if include_self:
        node = self
    else:
        node = self.parent
    while node is not None:
        yield node
        node = node.parent