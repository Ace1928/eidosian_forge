from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
@pytest.mark.parametrize('seed', [None, 42])
def test_with_spec_default_integer(seed):
    from dask.dataframe.io.demo import ColumnSpec, DatasetSpec, with_spec
    spec = DatasetSpec(npartitions=1, nrecords=5, column_specs=[ColumnSpec(dtype=int), ColumnSpec(dtype=int), ColumnSpec(dtype=int), ColumnSpec(dtype=int)])
    ddf = with_spec(spec, seed=seed)
    res = ddf.compute()
    for col in res.columns:
        assert 500 < res[col].min() < 1500
        assert 500 < res[col].max() < 1500