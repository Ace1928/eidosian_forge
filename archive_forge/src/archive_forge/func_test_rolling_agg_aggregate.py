from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def test_rolling_agg_aggregate():
    df = pd.DataFrame({'A': range(5), 'B': range(0, 10, 2)})
    ddf = dd.from_pandas(df, npartitions=3)
    assert_eq(df.rolling(window=3).agg(['mean', 'std']), ddf.rolling(window=3).agg(['mean', 'std']))
    assert_eq(df.rolling(window=3).agg({'A': 'sum', 'B': lambda x: np.std(x, ddof=1)}), ddf.rolling(window=3).agg({'A': 'sum', 'B': lambda x: np.std(x, ddof=1)}))
    assert_eq(df.rolling(window=3).agg(['sum', 'mean']), ddf.rolling(window=3).agg(['sum', 'mean']))
    assert_eq(df.rolling(window=3).agg({'A': ['sum', 'mean']}), ddf.rolling(window=3).agg({'A': ['sum', 'mean']}))
    kwargs = {'raw': True}
    assert_eq(df.rolling(window=3).apply(lambda x: np.std(x, ddof=1), **kwargs), ddf.rolling(window=3).apply(lambda x: np.std(x, ddof=1), **kwargs))