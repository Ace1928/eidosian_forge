from __future__ import annotations
import platform
from . import asyncio as _test_asyncio
from . import exclusions
from .exclusions import only_on
from .. import create_engine
from .. import util
from ..pool import QueuePool
@property
def standard_cursor_sql(self):
    """Target database passes SQL-92 style statements to cursor.execute()
        when a statement like select() or insert() is run.

        A very small portion of dialect-level tests will ensure that certain
        conditions are present in SQL strings, and these tests use very basic
        SQL that will work on any SQL-like platform in order to assert results.

        It's normally a given for any pep-249 DBAPI that a statement like
        "SELECT id, name FROM table WHERE some_table.id=5" will work.
        However, there are dialects that don't actually produce SQL Strings
        and instead may work with symbolic objects instead, or dialects that
        aren't working with SQL, so for those this requirement can be marked
        as excluded.

        """
    return exclusions.open()