from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_meta(db):
    data = read_sql_table('test', db, index_col='number', meta=dd.from_pandas(df, npartitions=1)).compute()
    assert (data.name == df.name).all()
    assert data.index.name == 'number'
    assert_eq(data, df)