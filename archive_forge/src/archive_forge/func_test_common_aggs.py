from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
@pytest.mark.parametrize('agg', ['nunique', 'mean', 'count', 'size', 'quantile'])
def test_common_aggs(agg):
    index = pd.date_range('2000-01-01', '2000-02-15', freq='h')
    ps = pd.Series(range(len(index)), index=index)
    ds = dd.from_pandas(ps, npartitions=2)
    f = lambda df: getattr(df, agg)()
    res = f(ps.resample('1d'))
    expected = f(ds.resample('1d'))
    assert_eq(res, expected, check_dtype=False)