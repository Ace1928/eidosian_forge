import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def copy_tree(self, from_relpath, to_relpath):
    """Copy a subtree from one relpath to another.

        If a faster implementation is available, specific transports should
        implement it.
        """
    source = self.clone(from_relpath)
    target = self.clone(to_relpath)
    stat = self.stat(from_relpath)
    target.mkdir('.', stat.st_mode & 511)
    source.copy_tree_to_transport(target)