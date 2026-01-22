from __future__ import unicode_literals
import os.path as op
from send2trash.compat import text_type
from send2trash.util import preprocess_paths
from ctypes import (
from ctypes.wintypes import HWND, UINT, LPCWSTR, BOOL
def prefix_and_path(path):
    """Guess the long-path prefix based on the kind of *path*.
    Local paths (C:\\folder\\file.ext) and UNC names (\\\\server\\folder\\file.ext)
    are handled.

    Return a tuple of the long-path prefix and the prefixed path.
    """
    prefix, long_path = ('\\\\?\\', path)
    if not path.startswith(prefix):
        if path.startswith('\\\\'):
            prefix = '\\\\?\\UNC'
            long_path = prefix + path[1:]
        else:
            long_path = prefix + path
    elif path.startswith(prefix + 'UNC\\'):
        prefix = '\\\\?\\UNC'
    return (prefix, long_path)