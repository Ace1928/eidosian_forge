import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
def reverse_iter(self):
    """Iterates over the direct children of this node (left->right)."""
    for c in reversed(self._children):
        yield c