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
@pytest.mark.parametrize('conn', all_connectable_types)
def test_api_date_parsing(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    df = sql.read_sql_query('SELECT * FROM types', conn)
    if not ('mysql' in conn_name or 'postgres' in conn_name):
        assert not issubclass(df.DateCol.dtype.type, np.datetime64)
    df = sql.read_sql_query('SELECT * FROM types', conn, parse_dates=['DateCol'])
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [Timestamp(2000, 1, 3, 0, 0, 0), Timestamp(2000, 1, 4, 0, 0, 0)]
    df = sql.read_sql_query('SELECT * FROM types', conn, parse_dates={'DateCol': '%Y-%m-%d %H:%M:%S'})
    assert issubclass(df.DateCol.dtype.type, np.datetime64)
    assert df.DateCol.tolist() == [Timestamp(2000, 1, 3, 0, 0, 0), Timestamp(2000, 1, 4, 0, 0, 0)]
    df = sql.read_sql_query('SELECT * FROM types', conn, parse_dates=['IntDateCol'])
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp(1986, 12, 25, 0, 0, 0), Timestamp(2013, 1, 1, 0, 0, 0)]
    df = sql.read_sql_query('SELECT * FROM types', conn, parse_dates={'IntDateCol': 's'})
    assert issubclass(df.IntDateCol.dtype.type, np.datetime64)
    assert df.IntDateCol.tolist() == [Timestamp(1986, 12, 25, 0, 0, 0), Timestamp(2013, 1, 1, 0, 0, 0)]
    df = sql.read_sql_query('SELECT * FROM types', conn, parse_dates={'IntDateOnlyCol': '%Y%m%d'})
    assert issubclass(df.IntDateOnlyCol.dtype.type, np.datetime64)
    assert df.IntDateOnlyCol.tolist() == [Timestamp('2010-10-10'), Timestamp('2010-12-12')]