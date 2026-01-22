from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('window', ['2s', '5s', '20s', '10h'])
def test_time_rolling_large_window_variable_chunks(window):
    df = pd.DataFrame({'a': pd.date_range('2016-01-01 00:00:00', periods=100, freq='1s'), 'b': np.random.randint(100, size=(100,))})
    ddf = dd.from_pandas(df, 5)
    ddf = ddf.repartition(divisions=[0, 5, 20, 28, 33, 54, 79, 80, 82, 99])
    df = df.set_index('a')
    ddf = ddf.set_index('a')
    assert_eq(ddf.rolling(window).sum(), df.rolling(window).sum())
    assert_eq(ddf.rolling(window).count(), df.rolling(window).count())
    assert_eq(ddf.rolling(window).mean(), df.rolling(window).mean())