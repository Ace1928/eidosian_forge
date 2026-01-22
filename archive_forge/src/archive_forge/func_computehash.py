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
def computehash(self, hashtype='md5', chunksize=524288):
    """Return hexdigest of hashvalue for this file."""
    try:
        try:
            import hashlib as mod
        except ImportError:
            if hashtype == 'sha1':
                hashtype = 'sha'
            mod = __import__(hashtype)
        hash = getattr(mod, hashtype)()
    except (AttributeError, ImportError):
        raise ValueError(f"Don't know how to compute {hashtype!r} hash")
    f = self.open('rb')
    try:
        while 1:
            buf = f.read(chunksize)
            if not buf:
                return hash.hexdigest()
            hash.update(buf)
    finally:
        f.close()