from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_datetimes():
    import datetime
    now = datetime.datetime.now()
    d = datetime.timedelta(seconds=1)
    df = pd.DataFrame({'a': list('ghjkl'), 'b': [now + i * d for i in range(2, -3, -1)]})
    with tmpfile() as f:
        uri = 'sqlite:///%s' % f
        df.to_sql('test', uri, index=False, if_exists='replace')
        data = read_sql_table('test', uri, npartitions=2, index_col='b')
        assert data.index.dtype.kind == 'M'
        assert data.divisions[0] == df.b.min()
        df2 = df.set_index('b')
        assert_eq(data.map_partitions(lambda x: x.sort_index()), df2.sort_index())