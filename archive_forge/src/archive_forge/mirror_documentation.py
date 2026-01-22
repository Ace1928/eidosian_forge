from __future__ import print_function, unicode_literals
import typing
from ._bulk import Copier
from .copy import copy_file_internal
from .errors import ResourceNotFound
from .opener import manage_fs
from .tools import is_thread_safe
from .walk import Walker
Mirror files / directories from one filesystem to another.

    Mirroring a filesystem will create an exact copy of ``src_fs`` on
    ``dst_fs``, by removing any files / directories on the destination
    that aren't on the source, and copying files that aren't.

    Arguments:
        src_fs (FS or str): Source filesystem (URL or instance).
        dst_fs (FS or str): Destination filesystem (URL or instance).
        walker (~fs.walk.Walker, optional): An optional walker instance.
        copy_if_newer (bool): Only copy newer files (the default).
        workers (int): Number of worker threads used
            (0 for single threaded). Set to a relatively low number
            for network filesystems, 4 would be a good start.
        preserve_time (bool): If `True`, try to preserve mtime of the
            resources (defaults to `False`).

    