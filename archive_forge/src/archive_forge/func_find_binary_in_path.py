import os
import sys
import asyncio
import threading
from uuid import uuid4
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from lazyops.models import LazyData
from lazyops.common import lazy_import, lazylibs
from lazyops.retry import retryable
def find_binary_in_path(filename):
    """
    Searches for a binary named `filename` in the current PATH. If an executable is found, its absolute path is returned
    else None.
    :param filename: Filename of the binary
    :return: Absolute path or None
    """
    if 'PATH' not in os.environ:
        return None
    for directory in os.environ['PATH'].split(get_variable_separator()):
        binary = os.path.abspath(os.path.join(directory, filename))
        if os.path.isfile(binary) and os.access(binary, os.X_OK):
            return binary
    return None