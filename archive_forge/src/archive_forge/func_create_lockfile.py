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
def create_lockfile(path):
    """Exclusively create lockfile. Throws when failed"""
    mypid = os.getpid()
    lockfile = path.join('.lock')
    if hasattr(lockfile, 'mksymlinkto'):
        lockfile.mksymlinkto(str(mypid))
    else:
        fd = error.checked_call(os.open, str(lockfile), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 420)
        with os.fdopen(fd, 'w') as f:
            f.write(str(mypid))
    return lockfile