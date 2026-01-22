import collections
import io
import itertools
import os
from taskflow.types import graph
from taskflow.utils import iter_utils
from taskflow.utils import misc
def child_count(self, only_direct=True):
    """Returns how many children this node has.

        This can be either only the direct children of this node or inclusive
        of all children nodes of this node (children of children and so-on).

        NOTE(harlowja): it does not account for the current node in this count.
        """
    if not only_direct:
        return iter_utils.count(self.dfs_iter())
    return len(self._children)