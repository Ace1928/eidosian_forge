from __future__ import unicode_literals
import typing
from collections import defaultdict, deque, namedtuple
from ._repr import make_repr
from .errors import FSError
from .path import abspath, combine, normpath
def check_scan_dir(self, fs, path, info):
    """Check if a directory should be scanned.

        Override to omit scanning of certain directories. If a directory
        is omitted, it will appear in the walk but its files and
        sub-directories will not.

        Arguments:
            fs (FS): A filesystem instance.
            path (str): Path to directory.
            info (Info): A resource info object for the directory.

        Returns:
            bool: `True` if the directory should be scanned.

        """
    return True