import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def copy_tree_to_transport(self, to_transport):
    """Copy a subtree from one transport to another.

        self.base is used as the source tree root, and to_transport.base
        is used as the target.  to_transport.base must exist (and be a
        directory).
        """
    files = []
    directories = ['.']
    while directories:
        dir = directories.pop()
        if dir != '.':
            to_transport.mkdir(dir)
        for path in self.list_dir(dir):
            path = dir + '/' + path
            stat = self.stat(path)
            if S_ISDIR(stat.st_mode):
                directories.append(path)
            else:
                files.append(path)
    self.copy_to(files, to_transport)