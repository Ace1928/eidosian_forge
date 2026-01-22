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
@pytest.mark.parametrize('conn', all_connectable)
@pytest.mark.parametrize('dtype', [None, int, float, {'A': int, 'B': float}])
def test_api_dtype_argument(conn, request, dtype):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_dtype_argument', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_dtype_argument')
    df = DataFrame([[1.2, 3.4], [5.6, 7.8]], columns=['A', 'B'])
    assert df.to_sql(name='test_dtype_argument', con=conn) == 2
    expected = df.astype(dtype)
    if 'postgres' in conn_name:
        query = 'SELECT "A", "B" FROM test_dtype_argument'
    else:
        query = 'SELECT A, B FROM test_dtype_argument'
    result = sql.read_sql_query(query, con=conn, dtype=dtype)
    tm.assert_frame_equal(result, expected)