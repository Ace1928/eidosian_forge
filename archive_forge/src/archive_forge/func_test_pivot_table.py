from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from packaging.version import parse as parse_version
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_VERSION, tm
from dask.dataframe.reshape import _get_dummies_dtype_default
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('values', ['B', ['B'], ['B', 'D']])
@pytest.mark.parametrize('aggfunc', ['mean', 'sum', 'count', 'first', 'last'])
def test_pivot_table(values, aggfunc):
    df = pd.DataFrame({'A': np.random.choice(list('XYZ'), size=100), 'B': np.random.randn(100), 'C': pd.Categorical(np.random.choice(list('abc'), size=100)), 'D': np.random.randn(100)})
    ddf = dd.from_pandas(df, 5).repartition((0, 20, 40, 60, 80, 98, 99))
    res = dd.pivot_table(ddf, index='A', columns='C', values=values, aggfunc=aggfunc)
    exp = pd.pivot_table(df, index='A', columns='C', values=values, aggfunc=aggfunc, observed=False)
    if aggfunc == 'count':
        exp = exp.astype(np.float64)
    assert_eq(res, exp)
    res = ddf.pivot_table(index='A', columns='C', values=values, aggfunc=aggfunc)
    exp = df.pivot_table(index='A', columns='C', values=values, aggfunc=aggfunc, observed=False)
    if aggfunc == 'count':
        exp = exp.astype(np.float64)
    assert_eq(res, exp)