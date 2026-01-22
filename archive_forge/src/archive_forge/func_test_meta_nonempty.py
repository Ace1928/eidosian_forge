from __future__ import annotations
import re
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_300, tm
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.utils import (
from dask.local import get_sync
def test_meta_nonempty():
    df1 = pd.DataFrame({'A': pd.Categorical(['Alice', 'Bob', 'Carol']), 'B': list('abc'), 'C': 'bar', 'D': np.float32(1), 'E': np.int32(1), 'F': pd.Timestamp('2016-01-01'), 'G': pd.date_range('2016-01-01', periods=3, tz='America/New_York'), 'H': pd.Timedelta('1 hours'), 'I': np.void(b' '), 'J': pd.Categorical([UNKNOWN_CATEGORIES] * 3), 'K': pd.Categorical([None, None, None])}, columns=list('DCBAHGFEIJK'))
    df2 = df1.iloc[0:0]
    df3 = meta_nonempty(df2)
    assert (df3.dtypes == df2.dtypes).all()
    assert df3['A'][0] == 'Alice'
    assert df3['B'][0] == 'foo'
    assert df3['C'][0] == 'foo'
    assert df3['D'][0] == np.float32(1)
    assert df3['D'][0].dtype == 'f4'
    assert df3['E'][0] == np.int32(1)
    assert df3['E'][0].dtype == 'i4'
    assert df3['F'][0] == pd.Timestamp('1970-01-01 00:00:00')
    assert df3['G'][0] == pd.Timestamp('1970-01-01 00:00:00', tz='America/New_York')
    assert df3['H'][0] == pd.Timedelta('1')
    assert df3['I'][0] == 'foo'
    assert df3['J'][0] == UNKNOWN_CATEGORIES
    assert len(df3['K'].cat.categories) == 0
    s = meta_nonempty(df2['A'])
    assert s.dtype == df2['A'].dtype
    assert (df3['A'] == s).all()