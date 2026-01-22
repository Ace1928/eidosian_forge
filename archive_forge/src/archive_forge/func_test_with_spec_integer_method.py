from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_with_spec_integer_method():
    from dask.dataframe.io.demo import ColumnSpec, DatasetSpec, with_spec
    spec = DatasetSpec(npartitions=1, nrecords=5, column_specs=[ColumnSpec(prefix='pois', dtype=int, method='poisson'), ColumnSpec(prefix='norm', dtype=int, method='normal'), ColumnSpec(prefix='unif', dtype=int, method='uniform'), ColumnSpec(prefix='binom', dtype=int, method='binomial', args=(100, 0.4)), ColumnSpec(prefix='choice', dtype=int, method='choice', args=(10,)), ColumnSpec(prefix='rand', dtype=int, random=True, low=0, high=10), ColumnSpec(prefix='rand', dtype=int, random=True)])
    ddf = with_spec(spec, seed=42)
    res = ddf.compute()
    assert res['pois1'].tolist() == [1002, 985, 947, 1003, 1017]
    assert res['norm1'].tolist() == [-1097, -276, 853, 272, 784]
    assert res['unif1'].tolist() == [772, 972, 798, 393, 656]
    assert res['binom1'].tolist() == [34, 46, 38, 37, 43]
    assert res['choice1'].tolist() == [0, 3, 1, 6, 6]
    assert res['rand1'].tolist() == [4, 6, 9, 4, 5]
    assert res['rand2'].tolist() == [883, 104, 192, 648, 256]