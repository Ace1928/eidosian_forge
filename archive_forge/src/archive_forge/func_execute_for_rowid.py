from __future__ import annotations
import contextlib
import re
import sqlite3
from typing import cast, Any, Iterable, Iterator, Tuple
from coverage.debug import auto_repr, clipped_repr, exc_one_line
from coverage.exceptions import DataError
from coverage.types import TDebugCtl
def execute_for_rowid(self, sql: str, parameters: Iterable[Any]=()) -> int:
    """Like execute, but returns the lastrowid."""
    with self.execute(sql, parameters) as cur:
        assert cur.lastrowid is not None
        rowid: int = cur.lastrowid
    if self.debug.should('sqldata'):
        self.debug.write(f'Row id result: {rowid!r}')
    return rowid