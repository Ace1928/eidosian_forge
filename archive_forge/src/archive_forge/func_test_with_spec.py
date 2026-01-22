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
def test_with_spec(seed):
    """Make a dataset with default random columns"""
    from dask.dataframe.io.demo import DatasetSpec, with_spec
    spec = DatasetSpec(nrecords=10, npartitions=2)
    ddf = with_spec(spec, seed=seed)
    assert isinstance(ddf, dd.DataFrame)
    assert ddf.npartitions == 2
    assert ddf.columns.tolist() == ['i1', 'f1', 'c1', 's1']
    assert ddf['i1'].dtype == 'int64'
    assert ddf['f1'].dtype == float
    assert ddf['c1'].dtype.name == 'category'
    assert ddf['s1'].dtype == get_string_dtype() if not dd._dask_expr_enabled() else object
    res = ddf.compute()
    assert len(res) == 10