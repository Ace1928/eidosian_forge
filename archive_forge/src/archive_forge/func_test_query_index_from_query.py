from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_query_index_from_query(db):
    from sqlalchemy import sql
    number = sql.column('number')
    name = sql.column('name')
    s1 = sql.select(number, name, sql.func.length(name).label('lenname')).select_from(sql.table('test'))
    out = read_sql_query(s1, db, npartitions=2, index_col='lenname')
    lenname_df = df.copy()
    lenname_df['lenname'] = lenname_df['name'].str.len()
    lenname_df = lenname_df.reset_index().set_index('lenname')
    assert_eq(out, lenname_df.loc[:, ['number', 'name']])