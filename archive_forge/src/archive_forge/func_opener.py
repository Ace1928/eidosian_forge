import functools as _functools
import warnings as _warnings
import io as _io
import os as _os
import shutil as _shutil
import stat as _stat
import errno as _errno
from random import Random as _Random
import sys as _sys
import types as _types
import weakref as _weakref
import _thread
def opener(*args):
    nonlocal fd
    flags2 = (flags | _os.O_TMPFILE) & ~_os.O_CREAT
    fd = _os.open(dir, flags2, 384)
    return fd