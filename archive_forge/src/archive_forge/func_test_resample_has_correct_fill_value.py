from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('method', ['count', 'nunique', 'size', 'sum'])
def test_resample_has_correct_fill_value(method):
    index = pd.date_range('2000-01-01', '2000-02-15', freq='h')
    index = index.union(pd.date_range('4-15-2000', '5-15-2000', freq='h'))
    ps = pd.Series(range(len(index)), index=index)
    ds = dd.from_pandas(ps, npartitions=2)
    assert_eq(getattr(ds.resample('30min'), method)(), getattr(ps.resample('30min'), method)())