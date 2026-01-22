from __future__ import annotations
import contextlib
import re
import sqlite3
from typing import cast, Any, Iterable, Iterator, Tuple
from coverage.debug import auto_repr, clipped_repr, exc_one_line
from coverage.exceptions import DataError
from coverage.types import TDebugCtl
def executemany_void(self, sql: str, data: Iterable[Any]) -> None:
    """Same as :meth:`python:sqlite3.Connection.executemany` when you don't need the cursor."""
    data = list(data)
    if data:
        self._executemany(sql, data).close()