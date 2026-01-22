from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import (
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid
import numpy as np
import pytest
from pandas._libs import lib
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
def test_not_reflect_all_tables(sqlite_conn):
    conn = sqlite_conn
    from sqlalchemy import text
    from sqlalchemy.engine import Engine
    query_list = [text('CREATE TABLE invalid (x INTEGER, y UNKNOWN);'), text('CREATE TABLE other_table (x INTEGER, y INTEGER);')]
    for query in query_list:
        if isinstance(conn, Engine):
            with conn.connect() as conn:
                with conn.begin():
                    conn.execute(query)
        else:
            with conn.begin():
                conn.execute(query)
    with tm.assert_produces_warning(None):
        sql.read_sql_table('other_table', conn)
        sql.read_sql_query('SELECT * FROM other_table', conn)