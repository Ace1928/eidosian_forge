from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def copy_file_if(src_fs, src_path, dst_fs, dst_path, condition, preserve_time=False):
    """Copy a file from one filesystem to another, depending on a condition.

    Depending on the value of ``condition``, certain requirements must
    be fulfilled for a file to be copied to ``dst_fs``. The following
    values are supported:

    ``"always"``
        The source file is always copied.
    ``"newer"``
        The last modification time of the source file must be newer than that
        of the destination file. If either file has no modification time, the
        copy is performed always.
    ``"older"``
        The last modification time of the source file must be older than that
        of the destination file. If either file has no modification time, the
        copy is performed always.
    ``"exists"``
        The source file is only copied if a file of the same path already
        exists in ``dst_fs``.
    ``"not_exists"``
        The source file is only copied if no file of the same path already
        exists in ``dst_fs``.

    Arguments:
        src_fs (FS or str): Source filesystem (instance or URL).
        src_path (str): Path to a file on the source filesystem.
        dst_fs (FS or str): Destination filesystem (instance or URL).
        dst_path (str): Path to a file on the destination filesystem.
        condition (str): Name of the condition to check for each file.
        preserve_time (bool): If `True`, try to preserve mtime of the
            resource (defaults to `False`).

    Returns:
        bool: `True` if the file copy was executed, `False` otherwise.

    """
    with manage_fs(src_fs, writeable=False) as _src_fs:
        with manage_fs(dst_fs, create=True) as _dst_fs:
            do_copy = _copy_is_necessary(_src_fs, src_path, _dst_fs, dst_path, condition)
            if do_copy:
                copy_file_internal(_src_fs, src_path, _dst_fs, dst_path, preserve_time=preserve_time, lock=True)
            return do_copy