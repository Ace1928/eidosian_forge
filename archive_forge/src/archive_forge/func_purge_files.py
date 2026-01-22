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
def purge_files(self, filenames: Collection[str]) -> None:
    """Purge any existing coverage data for the given `filenames`.

        .. versionadded:: 7.2

        """
    if self._debug.should('dataop'):
        self._debug.write(f'Purging data for {filenames!r}')
    self._start_using()
    with self._connect() as con:
        if self._has_lines:
            sql = 'delete from line_bits where file_id=?'
        elif self._has_arcs:
            sql = 'delete from arc where file_id=?'
        else:
            raise DataError("Can't purge files in an empty CoverageData")
        for filename in filenames:
            file_id = self._file_id(filename, add=False)
            if file_id is None:
                continue
            con.execute_void(sql, (file_id,))