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
def test_double_precision(conn, request):
    if conn == 'sqlite_str':
        pytest.skip('sqlite_str has no inspection system')
    conn = request.getfixturevalue(conn)
    from sqlalchemy import BigInteger, Float, Integer
    from sqlalchemy.schema import MetaData
    V = 1.2345678910111213
    df = DataFrame({'f32': Series([V], dtype='float32'), 'f64': Series([V], dtype='float64'), 'f64_as_f32': Series([V], dtype='float64'), 'i32': Series([5], dtype='int32'), 'i64': Series([5], dtype='int64')})
    assert df.to_sql(name='test_dtypes', con=conn, index=False, if_exists='replace', dtype={'f64_as_f32': Float(precision=23)}) == 1
    res = sql.read_sql_table('test_dtypes', conn)
    assert np.round(df['f64'].iloc[0], 14) == np.round(res['f64'].iloc[0], 14)
    meta = MetaData()
    meta.reflect(bind=conn)
    col_dict = meta.tables['test_dtypes'].columns
    assert str(col_dict['f32'].type) == str(col_dict['f64_as_f32'].type)
    assert isinstance(col_dict['f32'].type, Float)
    assert isinstance(col_dict['f64'].type, Float)
    assert isinstance(col_dict['i32'].type, Integer)
    assert isinstance(col_dict['i64'].type, BigInteger)