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
def test_api_read_sql_duplicate_columns(conn, request):
    if 'adbc' in conn:
        request.node.add_marker(pytest.mark.xfail(reason='pyarrow->pandas throws ValueError', strict=True))
    conn = request.getfixturevalue(conn)
    if sql.has_table('test_table', conn):
        with sql.SQLDatabase(conn, need_transaction=True) as pandasSQL:
            pandasSQL.drop_table('test_table')
    df = DataFrame({'a': [1, 2, 3], 'b': [0.1, 0.2, 0.3], 'c': 1})
    df.to_sql(name='test_table', con=conn, index=False)
    result = pd.read_sql('SELECT a, b, a +1 as a, c FROM test_table', conn)
    expected = DataFrame([[1, 0.1, 2, 1], [2, 0.2, 3, 1], [3, 0.3, 4, 1]], columns=['a', 'b', 'a', 'c'])
    tm.assert_frame_equal(result, expected)