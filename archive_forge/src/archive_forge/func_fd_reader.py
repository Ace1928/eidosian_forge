from contextlib import contextmanager
from ctypes import cast, c_void_p, POINTER, create_string_buffer
from os import fstat, stat
from . import ffi
from .ffi import (
from .entry import ArchiveEntry, PassedArchiveEntry
@contextmanager
def fd_reader(fd, format_name='all', filter_name='all', block_size=4096, passphrase=None, header_codec='utf-8'):
    """Read an archive from a file descriptor.
    """
    with new_archive_read(format_name, filter_name, passphrase) as archive_p:
        try:
            block_size = fstat(fd).st_blksize
        except (OSError, AttributeError):
            pass
        ffi.read_open_fd(archive_p, fd, block_size)
        yield ArchiveRead(archive_p, header_codec)