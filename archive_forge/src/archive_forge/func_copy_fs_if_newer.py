from __future__ import print_function, unicode_literals
import typing
import warnings
from .errors import ResourceNotFound
from .opener import manage_fs
from .path import abspath, combine, frombase, normpath
from .tools import is_thread_safe
from .walk import Walker
def copy_fs_if_newer(src_fs, dst_fs, walker=None, on_copy=None, workers=0, preserve_time=False):
    """Copy the contents of one filesystem to another, checking times.

    .. deprecated:: 2.5.0
       Use `~fs.copy.copy_fs_if` with ``condition="newer"`` instead.

    """
    warnings.warn('copy_fs_if_newer is deprecated. Use copy_fs_if instead.', DeprecationWarning)
    return copy_fs_if(src_fs, dst_fs, 'newer', walker, on_copy, workers, preserve_time=preserve_time)