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
def test_api_chunksize_read(conn, request):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='chunksize argument NotImplemented with ADBC'))
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_chunksize', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_chunksize')
    df = DataFrame(np.random.default_rng(2).standard_normal((22, 5)), columns=list('abcde'))
    df.to_sql(name='test_chunksize', con=conn, index=False)
    res1 = sql.read_sql_query('select * from test_chunksize', conn)
    res2 = DataFrame()
    i = 0
    sizes = [5, 5, 5, 5, 2]
    for chunk in sql.read_sql_query('select * from test_chunksize', conn, chunksize=5):
        res2 = concat([res2, chunk], ignore_index=True)
        assert len(chunk) == sizes[i]
        i += 1
    tm.assert_frame_equal(res1, res2)
    if conn_name == 'sqlite_buildin':
        with pytest.raises(NotImplementedError, match=''):
            sql.read_sql_table('test_chunksize', conn, chunksize=5)
    else:
        res3 = DataFrame()
        i = 0
        sizes = [5, 5, 5, 5, 2]
        for chunk in sql.read_sql_table('test_chunksize', conn, chunksize=5):
            res3 = concat([res3, chunk], ignore_index=True)
            assert len(chunk) == sizes[i]
            i += 1
        tm.assert_frame_equal(res1, res3)