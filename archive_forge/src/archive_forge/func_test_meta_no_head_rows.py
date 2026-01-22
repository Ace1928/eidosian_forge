from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_meta_no_head_rows(db):
    data = read_sql_table('test', db, index_col='number', meta=dd.from_pandas(df, npartitions=1), npartitions=2, head_rows=0)
    assert len(data.divisions) == 3
    data = data.compute()
    assert (data.name == df.name).all()
    assert data.index.name == 'number'
    assert_eq(data, df)
    data = read_sql_table('test', db, index_col='number', meta=dd.from_pandas(df, npartitions=1), divisions=[0, 3, 6], head_rows=0)
    assert len(data.divisions) == 3
    data = data.compute()
    assert (data.name == df.name).all()
    assert data.index.name == 'number'
    assert_eq(data, df)