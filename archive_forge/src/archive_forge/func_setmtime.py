from __future__ import annotations
import atexit
from contextlib import contextmanager
import fnmatch
import importlib.util
import io
import os
from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import isabs
from os.path import isdir
from os.path import isfile
from os.path import islink
from os.path import normpath
import posixpath
from stat import S_ISDIR
from stat import S_ISLNK
from stat import S_ISREG
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
from typing import overload
from typing import TYPE_CHECKING
import uuid
import warnings
from . import error
def setmtime(self, mtime=None):
    """Set modification time for the given path.  if 'mtime' is None
        (the default) then the file's mtime is set to current time.

        Note that the resolution for 'mtime' is platform dependent.
        """
    if mtime is None:
        return error.checked_call(os.utime, self.strpath, mtime)
    try:
        return error.checked_call(os.utime, self.strpath, (-1, mtime))
    except error.EINVAL:
        return error.checked_call(os.utime, self.strpath, (self.atime(), mtime))