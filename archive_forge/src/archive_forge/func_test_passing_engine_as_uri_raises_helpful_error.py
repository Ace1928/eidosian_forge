from __future__ import annotations
import io
import sys
from contextlib import contextmanager
import pytest
from dask.dataframe.io.sql import read_sql, read_sql_query, read_sql_table
from dask.dataframe.utils import assert_eq, get_string_dtype
from dask.utils import tmpfile
def test_passing_engine_as_uri_raises_helpful_error(db):
    from sqlalchemy import create_engine
    df = pd.DataFrame([{'i': i, 's': str(i) * 2} for i in range(4)])
    ddf = dd.from_pandas(df, npartitions=2)
    with tmpfile() as f:
        db = 'sqlite:///%s' % f
        engine = create_engine(db)
        with pytest.raises(ValueError, match='Expected URI to be a string'):
            ddf.to_sql('test', engine, if_exists='replace')