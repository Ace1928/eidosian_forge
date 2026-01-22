from contextlib import contextmanager
from ctypes import byref, cast, c_char, c_size_t, c_void_p, POINTER
from posixpath import join
import warnings
from . import ffi
from .entry import ArchiveEntry, FileType
from .ffi import (
def write_cb_internal(archive_p, context, buffer_, length):
    data = cast(buffer_, POINTER(c_char * length))[0]
    return write_func(data)