from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def copy_dir_if_newer(src_fs, src_path, dst_fs, dst_path, walker=None, on_copy=None, workers=0, preserve_time=False):
    """Copy a directory from one filesystem to another, checking times.

    .. deprecated:: 2.5.0
       Use `~fs.copy.copy_dir_if` with ``condition="newer"`` instead.

    """
    warnings.warn('copy_dir_if_newer is deprecated. Use copy_dir_if instead.', DeprecationWarning)
    copy_dir_if(src_fs, src_path, dst_fs, dst_path, 'newer', walker, on_copy, workers, preserve_time=preserve_time)