from contextlib import contextmanager
from ctypes import cast, c_void_p, POINTER, create_string_buffer
from os import fstat, stat
from . import ffi
from .ffi import (
from .entry import ArchiveEntry, PassedArchiveEntry
@contextmanager
def memory_reader(buf, format_name='all', filter_name='all', passphrase=None, header_codec='utf-8'):
    """Read an archive from memory.
    """
    with new_archive_read(format_name, filter_name, passphrase) as archive_p:
        ffi.read_open_memory(archive_p, cast(buf, c_void_p), len(buf))
        yield ArchiveRead(archive_p, header_codec)