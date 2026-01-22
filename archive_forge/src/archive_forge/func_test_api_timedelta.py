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
def test_api_timedelta(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_timedelta', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_timedelta')
    df = to_timedelta(Series(['00:00:01', '00:00:03'], name='foo')).to_frame()
    if conn_name == 'sqlite_adbc_conn':
        request.node.add_marker(pytest.mark.xfail(reason="sqlite ADBC driver doesn't implement timedelta"))
    if 'adbc' in conn_name:
        if pa_version_under14p1:
            exp_warning = DeprecationWarning
        else:
            exp_warning = None
    else:
        exp_warning = UserWarning
    with tm.assert_produces_warning(exp_warning, check_stacklevel=False):
        result_count = df.to_sql(name='test_timedelta', con=conn)
    assert result_count == 2
    result = sql.read_sql_query('SELECT * FROM test_timedelta', conn)
    if conn_name == 'postgresql_adbc_conn':
        expected = Series([pd.DateOffset(months=0, days=0, microseconds=1000000, nanoseconds=0), pd.DateOffset(months=0, days=0, microseconds=3000000, nanoseconds=0)], name='foo')
    else:
        expected = df['foo'].astype('int64')
    tm.assert_series_equal(result['foo'], expected)