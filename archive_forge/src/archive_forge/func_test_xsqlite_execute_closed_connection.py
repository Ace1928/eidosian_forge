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
def test_xsqlite_execute_closed_connection():
    create_sql = '\n    CREATE TABLE test\n    (\n    a TEXT,\n    b TEXT,\n    c REAL,\n    PRIMARY KEY (a, b)\n    );\n    '
    with contextlib.closing(sqlite3.connect(':memory:')) as conn:
        cur = conn.cursor()
        cur.execute(create_sql)
        with sql.pandasSQL_builder(conn) as pandas_sql:
            pandas_sql.execute('INSERT INTO test VALUES("foo", "bar", 1.234)')
    msg = 'Cannot operate on a closed database.'
    with pytest.raises(sqlite3.ProgrammingError, match=msg):
        tquery('select * from test', con=conn)