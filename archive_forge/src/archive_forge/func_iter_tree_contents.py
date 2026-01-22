import os
import stat
import sys
import warnings
from contextlib import suppress
from io import BytesIO
from typing import (
from .errors import NotTreeError
from .file import GitFile
from .objects import (
from .pack import (
from .protocol import DEPTH_INFINITE
from .refs import PEELED_TAG_SUFFIX, Ref
def iter_tree_contents(self, tree_id, include_trees=False):
    """Iterate the contents of a tree and all subtrees.

        Iteration is depth-first pre-order, as in e.g. os.walk.

        Args:
          tree_id: SHA1 of the tree.
          include_trees: If True, include tree objects in the iteration.
        Returns: Iterator over TreeEntry namedtuples for all the objects in a
            tree.
        """
    warnings.warn('Please use dulwich.object_store.iter_tree_contents', DeprecationWarning, stacklevel=2)
    return iter_tree_contents(self, tree_id, include_trees=include_trees)