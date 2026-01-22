import errno
import sys
from io import BytesIO
from stat import S_ISDIR
from typing import Any, Callable, Dict, TypeVar
from .. import errors, hooks, osutils, registry, ui, urlutils
from ..trace import mutter
def delete_tree(self, relpath):
    """Delete an entire tree. This may require a listable transport."""
    subtree = self.clone(relpath)
    files = []
    directories = ['.']
    pending_rmdirs = []
    while directories:
        dir = directories.pop()
        if dir != '.':
            pending_rmdirs.append(dir)
        for path in subtree.list_dir(dir):
            path = dir + '/' + path
            stat = subtree.stat(path)
            if S_ISDIR(stat.st_mode):
                directories.append(path)
            else:
                files.append(path)
    for file in files:
        subtree.delete(file)
    pending_rmdirs.reverse()
    for dir in pending_rmdirs:
        subtree.rmdir(dir)
    self.rmdir(relpath)