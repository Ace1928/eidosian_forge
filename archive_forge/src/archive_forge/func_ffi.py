from ctypes import (
import ctypes
from ctypes.util import find_library
import logging
import mmap
import os
import sysconfig
from .exception import ArchiveError
def ffi(name, argtypes, restype, errcheck=None):
    f = getattr(libarchive, 'archive_' + name)
    f.argtypes = argtypes
    f.restype = restype
    if errcheck:
        f.errcheck = errcheck
    globals()[name] = f
    return f