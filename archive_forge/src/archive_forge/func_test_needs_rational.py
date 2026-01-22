from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_needs_rational(db):
    import datetime
    now = datetime.datetime.now()
    d = datetime.timedelta(seconds=1)
    df = pd.DataFrame({'a': list('ghjkl'), 'b': [now + i * d for i in range(5)], 'c': [True, True, False, True, True]})
    df = pd.concat([df, pd.DataFrame([{'a': 'x', 'b': now + d * 1000, 'c': None}, {'a': None, 'b': now + d * 1001, 'c': None}])])
    string_dtype = get_string_dtype()
    with tmpfile() as f:
        uri = 'sqlite:///%s' % f
        df.to_sql('test', uri, index=False, if_exists='replace')
        data = read_sql_table('test', uri, npartitions=2, index_col='b')
        df2 = df.set_index('b')
        assert_eq(data, df2.astype({'c': bool}))
        data = read_sql_table('test', uri, npartitions=2, index_col='b', head_rows=12)
        df2 = df.set_index('b')
        assert_eq(data, df2)
        data = read_sql_table('test', uri, npartitions=20, index_col='b')
        part = data.get_partition(12).compute()
        assert part.dtypes.tolist() == [string_dtype, bool]
        assert part.empty
        df2 = df.set_index('b')
        assert_eq(data, df2.astype({'c': bool}))
        data = read_sql_table('test', uri, npartitions=2, index_col='b', meta=df2[:0])
        part = data.get_partition(1).compute()
        assert part.dtypes.tolist() == [string_dtype, string_dtype]
        df2 = df.set_index('b')
        assert_eq(data, df2)