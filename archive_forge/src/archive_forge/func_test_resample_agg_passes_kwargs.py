from __future__ import annotations
from itertools import product
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220
from dask.dataframe.utils import assert_eq
def test_resample_agg_passes_kwargs():
    index = pd.date_range('2000-01-01', '2000-02-15', freq='h')
    ps = pd.Series(range(len(index)), index=index)
    ds = dd.from_pandas(ps, npartitions=2)

    def foo(series, bar=1, *args, **kwargs):
        return bar
    assert_eq(ds.resample('2h').agg(foo, bar=2), ps.resample('2h').agg(foo, bar=2))
    assert (ds.resample('2h').agg(foo, bar=2) == 2).compute().all()