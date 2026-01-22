from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def copy_file_internal(src_fs, src_path, dst_fs, dst_path, preserve_time=False, lock=False):
    """Copy a file at low level, without calling `manage_fs` or locking.

    If the destination exists, and is a file, it will be first truncated.

    This method exists to optimize copying in loops. In general you
    should prefer `copy_file`.

    Arguments:
        src_fs (FS): Source filesystem.
        src_path (str): Path to a file on the source filesystem.
        dst_fs (FS): Destination filesystem.
        dst_path (str): Path to a file on the destination filesystem.
        preserve_time (bool): If `True`, try to preserve mtime of the
            resource (defaults to `False`).
        lock (bool): Lock both filesystems before copying.

    """
    if src_fs is dst_fs:
        src_fs.copy(src_path, dst_path, overwrite=True, preserve_time=preserve_time)
        return

    def _copy_locked():
        if dst_fs.hassyspath(dst_path):
            with dst_fs.openbin(dst_path, 'w') as write_file:
                src_fs.download(src_path, write_file)
        else:
            with src_fs.openbin(src_path) as read_file:
                dst_fs.upload(dst_path, read_file)
        if preserve_time:
            copy_modified_time(src_fs, src_path, dst_fs, dst_path)
    if lock:
        with src_fs.lock(), dst_fs.lock():
            _copy_locked()
    else:
        _copy_locked()