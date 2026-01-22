from __future__ import annotations
import contextlib
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import tokenize
from dask.dataframe._compat import PANDAS_GE_210, PANDAS_GE_220, IndexingError, tm
from dask.dataframe.indexing import _coerce_loc_index
from dask.dataframe.utils import assert_eq, make_meta, pyarrow_strings_enabled
def test_getitem_timestamp_str():
    df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)}, index=pd.date_range('2011-01-01', freq='h', periods=100))
    ddf = dd.from_pandas(df, 10)
    if not DASK_EXPR_ENABLED:
        with pytest.warns(FutureWarning, match='Indexing a DataFrame with a datetimelike'):
            assert_eq(df.loc['2011-01-02'], ddf['2011-01-02'])
    assert_eq(df['2011-01-02':'2011-01-10'], ddf['2011-01-02':'2011-01-10'])
    df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)}, index=pd.date_range('2011-01-01', freq='D', periods=100))
    ddf = dd.from_pandas(df, 50)
    assert_eq(df.loc['2011-01'], ddf.loc['2011-01'])
    assert_eq(df.loc['2011'], ddf.loc['2011'])
    assert_eq(df['2011-01':'2012-05'], ddf['2011-01':'2012-05'])
    assert_eq(df['2011':'2015'], ddf['2011':'2015'])