from contextlib import contextmanager
from ctypes import byref, cast, c_char, c_size_t, c_void_p, POINTER
from posixpath import join
import warnings
from . import ffi
from .entry import ArchiveEntry, FileType
from .ffi import (
@contextmanager
def new_archive_read_disk(path, flags=0, lookup=False):
    archive_p = read_disk_new()
    read_disk_set_behavior(archive_p, flags)
    if lookup:
        ffi.read_disk_set_standard_lookup(archive_p)
    read_disk_open_w(archive_p, path)
    try:
        yield archive_p
    finally:
        read_free(archive_p)