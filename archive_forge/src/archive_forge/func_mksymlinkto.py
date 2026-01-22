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
def mksymlinkto(self, value, absolute=1):
    """Create a symbolic link with the given value (pointing to another name)."""
    if absolute:
        error.checked_call(os.symlink, str(value), self.strpath)
    else:
        base = self.common(value)
        relsource = self.__class__(value).relto(base)
        reldest = self.relto(base)
        n = reldest.count(self.sep)
        target = self.sep.join(('..',) * n + (relsource,))
        error.checked_call(os.symlink, target, self.strpath)