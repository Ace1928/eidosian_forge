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
def measured_contexts(self) -> set[str]:
    """A set of all contexts that have been measured.

        .. versionadded:: 5.0

        """
    self._start_using()
    with self._connect() as con:
        with con.execute('select distinct(context) from context') as cur:
            contexts = {row[0] for row in cur}
    return contexts