from __future__ import annotations
import collections
import datetime
import functools
import glob
import itertools
import os
import random
import socket
import sqlite3
import string
import sys
import textwrap
import threading
import zlib
from typing import (
from coverage.debug import NoDebugging, auto_repr
from coverage.exceptions import CoverageException, DataError
from coverage.files import PathAliases
from coverage.misc import file_be_gone, isolate_module
from coverage.numbits import numbits_to_nums, numbits_union, nums_to_numbits
from coverage.sqlitedb import SqliteDb
from coverage.types import AnyCallable, FilePath, TArc, TDebugCtl, TLineNo, TWarnFn
from coverage.version import __version__
def filename_suffix(suffix: str | bool | None) -> str | None:
    """Compute a filename suffix for a data file.

    If `suffix` is a string or None, simply return it. If `suffix` is True,
    then build a suffix incorporating the hostname, process id, and a random
    number.

    Returns a string or None.

    """
    if suffix is True:
        die = random.Random(os.urandom(8))
        letters = string.ascii_uppercase + string.ascii_lowercase
        rolls = ''.join((die.choice(letters) for _ in range(6)))
        suffix = f'{socket.gethostname()}.{os.getpid()}.X{rolls}x'
    elif suffix is False:
        suffix = None
    return suffix