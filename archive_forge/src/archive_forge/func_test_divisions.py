from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_divisions(db):
    data = read_sql_table('test', db, columns=['name'], divisions=[0, 2, 4], index_col='number')
    assert data.divisions == (0, 2, 4)
    assert data.index.max().compute() == 4
    assert_eq(data, df[['name']][df.index <= 4])