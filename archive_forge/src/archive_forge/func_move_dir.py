from __future__ import print_function, unicode_literals
import typing
from ._pathcompat import commonpath
from .copy import copy_dir, copy_file
from .errors import FSError
from .opener import manage_fs
from .osfs import OSFS
from .path import frombase
def move_dir(src_fs, src_path, dst_fs, dst_path, workers=0, preserve_time=False):
    """Move a directory from one filesystem to another.

    Arguments:
        src_fs (FS or str): Source filesystem (instance or URL).
        src_path (str): Path to a directory on ``src_fs``
        dst_fs (FS or str): Destination filesystem (instance or URL).
        dst_path (str): Path to a directory on ``dst_fs``.
        workers (int): Use ``worker`` threads to copy data, or ``0``
            (default) for a single-threaded copy.
        preserve_time (bool): If `True`, try to preserve mtime of the
            resources (defaults to `False`).

    """
    with manage_fs(src_fs, writeable=True) as _src_fs:
        with manage_fs(dst_fs, writeable=True, create=True) as _dst_fs:
            with _src_fs.lock(), _dst_fs.lock():
                _dst_fs.makedir(dst_path, recreate=True)
                copy_dir(src_fs, src_path, dst_fs, dst_path, workers=workers, preserve_time=preserve_time)
                _src_fs.removetree(src_path)