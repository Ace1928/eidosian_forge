from __future__ import print_function, unicode_literals
import typing
from . import errors
from .errors import DirectoryNotEmpty, ResourceNotFound
from .path import abspath, dirname, normpath, recursepath
def copy_file_data(src_file, dst_file, chunk_size=None):
    """Copy data from one file object to another.

    Arguments:
        src_file (io.IOBase): File open for reading.
        dst_file (io.IOBase): File open for writing.
        chunk_size (int): Number of bytes to copy at
            a time (or `None` to use sensible default).

    """
    _chunk_size = 1024 * 1024 if chunk_size is None else chunk_size
    read = src_file.read
    write = dst_file.write
    for chunk in iter(lambda: read(_chunk_size) or None, None):
        write(chunk)