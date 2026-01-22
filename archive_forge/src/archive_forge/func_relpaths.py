import os
from typing import Set
from pyinotify import (IN_ATTRIB, IN_CLOSE_WRITE, IN_CREATE, IN_DELETE,
from .workingtree import WorkingTree
def relpaths(self) -> Set[str]:
    """Return the paths relative to the tree root that changed."""
    return {self._tree.relpath(p) for p in self.paths()}