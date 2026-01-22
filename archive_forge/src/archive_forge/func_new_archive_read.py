from contextlib import contextmanager
from ctypes import cast, c_void_p, POINTER, create_string_buffer
from os import fstat, stat
from . import ffi
from .ffi import (
from .entry import ArchiveEntry, PassedArchiveEntry
@contextmanager
def new_archive_read(format_name='all', filter_name='all', passphrase=None):
    """Creates an archive struct suitable for reading from an archive.

    Returns a pointer if successful. Raises ArchiveError on error.
    """
    archive_p = ffi.read_new()
    try:
        if passphrase:
            if not isinstance(passphrase, bytes):
                passphrase = passphrase.encode('utf-8')
            try:
                ffi.read_add_passphrase(archive_p, passphrase)
            except AttributeError:
                raise NotImplementedError(f"the libarchive being used (version {ffi.version_number()}, path {ffi.libarchive_path}) doesn't support encryption")
        ffi.get_read_filter_function(filter_name)(archive_p)
        ffi.get_read_format_function(format_name)(archive_p)
        yield archive_p
    finally:
        ffi.read_free(archive_p)