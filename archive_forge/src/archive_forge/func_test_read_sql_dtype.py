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
@pytest.mark.parametrize('dtype_backend', [lib.no_default, 'numpy_nullable'])
@pytest.mark.parametrize('func', ['read_sql', 'read_sql_query'])
def test_read_sql_dtype(conn, request, func, dtype_backend):
    conn = request.getfixturevalue(conn)
    table = 'test'
    df = DataFrame({'a': [1, 2, 3], 'b': 5})
    df.to_sql(name=table, con=conn, index=False, if_exists='replace')
    result = getattr(pd, func)(f'Select * from {table}', conn, dtype={'a': np.float64}, dtype_backend=dtype_backend)
    expected = DataFrame({'a': Series([1, 2, 3], dtype=np.float64), 'b': Series([5, 5, 5], dtype='int64' if not dtype_backend == 'numpy_nullable' else 'Int64')})
    tm.assert_frame_equal(result, expected)