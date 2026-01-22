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
def set_query_context(self, context: str) -> None:
    """Set a context for subsequent querying.

        The next :meth:`lines`, :meth:`arcs`, or :meth:`contexts_by_lineno`
        calls will be limited to only one context.  `context` is a string which
        must match a context exactly.  If it does not, no exception is raised,
        but queries will return no data.

        .. versionadded:: 5.0

        """
    self._start_using()
    with self._connect() as con:
        with con.execute('select id from context where context = ?', (context,)) as cur:
            self._query_context_ids = [row[0] for row in cur.fetchall()]