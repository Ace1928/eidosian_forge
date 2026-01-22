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
def test_assert_eq_sorts():
    df = pd.DataFrame({'A': np.linspace(0, 1, 10), 'B': np.random.random(10)})
    df_s = df.sort_values('B')
    assert_eq(df, df_s)
    with pytest.raises(AssertionError):
        assert_eq(df, df_s, sort_results=False)
    df_sr = df_s.reset_index(drop=True)
    assert_eq(df, df_sr, check_index=False)
    with pytest.raises(AssertionError):
        assert_eq(df, df_sr)
    with pytest.raises(AssertionError):
        assert_eq(df, df_sr, check_index=False, sort_results=False)
    ddf = dd.from_pandas(df, npartitions=2)
    ddf_s = ddf.sort_values(['B'])
    assert_eq(df, ddf_s)
    with pytest.raises(AssertionError):
        assert_eq(df, ddf_s, sort_results=False)
    ddf_sr = ddf_s.reset_index(drop=True)
    assert_eq(df, ddf_sr, check_index=False)
    with pytest.raises(AssertionError):
        assert_eq(df, ddf_sr, check_index=False, sort_results=False)