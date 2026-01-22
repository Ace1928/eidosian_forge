from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_query_with_meta(db):
    from sqlalchemy import sql
    data = {'name': pd.Series([], name='name', dtype='str'), 'age': pd.Series([], name='age', dtype='int')}
    index = pd.Index([], name='number', dtype='int')
    meta = pd.DataFrame(data, index=index)
    s1 = sql.select(sql.column('number'), sql.column('name'), sql.column('age')).select_from(sql.table('test'))
    out = read_sql_query(s1, db, npartitions=2, index_col='number', meta=meta)
    assert_eq(out, df[['name', 'age']], check_dtype=sys.platform != 'win32')