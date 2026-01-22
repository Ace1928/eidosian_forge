from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_to_sql_kwargs():
    ddf = dd.from_pandas(df, 2)
    with tmp_db_uri() as uri:
        ddf.to_sql('test', uri, method='multi')
        with pytest.raises(TypeError, match="to_sql\\(\\) got an unexpected keyword argument 'unknown'"):
            ddf.to_sql('test', uri, unknown=None)