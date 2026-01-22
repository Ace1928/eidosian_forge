import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def canonic(self, filename):
    """Return canonical form of filename.

        For real filenames, the canonical form is a case-normalized (on
        case insensitive filesystems) absolute path.  'Filenames' with
        angle brackets, such as "<stdin>", generated in interactive
        mode, are returned unchanged.
        """
    if filename == '<' + filename[1:-1] + '>':
        return filename
    canonic = self.fncache.get(filename)
    if not canonic:
        canonic = os.path.abspath(filename)
        canonic = os.path.normcase(canonic)
        self.fncache[filename] = canonic
    return canonic