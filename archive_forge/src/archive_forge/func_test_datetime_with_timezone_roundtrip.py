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
@pytest.mark.parametrize('conn', sqlalchemy_connectable)
def test_datetime_with_timezone_roundtrip(conn, request):
    conn_name = conn
    conn = request.getfixturevalue(conn)
    expected = DataFrame({'A': date_range('2013-01-01 09:00:00', periods=3, tz='US/Pacific')})
    assert expected.to_sql(name='test_datetime_tz', con=conn, index=False) == 3
    if 'postgresql' in conn_name:
        expected['A'] = expected['A'].dt.tz_convert('UTC')
    else:
        expected['A'] = expected['A'].dt.tz_localize(None)
    result = sql.read_sql_table('test_datetime_tz', conn)
    tm.assert_frame_equal(result, expected)
    result = sql.read_sql_query('SELECT * FROM test_datetime_tz', conn)
    if 'sqlite' in conn_name:
        assert isinstance(result.loc[0, 'A'], str)
        result['A'] = to_datetime(result['A'])
    tm.assert_frame_equal(result, expected)