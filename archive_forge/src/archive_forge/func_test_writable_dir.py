import fnmatch
import os
import os.path
import random
import sys
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Generator, List, Union, cast
from pip._vendor.tenacity import retry, stop_after_delay, wait_fixed
from pip._internal.utils.compat import get_path_uid
from pip._internal.utils.misc import format_size
def test_writable_dir(path: str) -> bool:
    """Check if a directory is writable.

    Uses os.access() on POSIX, tries creating files on Windows.
    """
    while not os.path.isdir(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    if os.name == 'posix':
        return os.access(path, os.W_OK)
    return _test_writable_dir_win(path)