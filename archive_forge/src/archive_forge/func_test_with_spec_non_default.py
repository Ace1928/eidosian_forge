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
def test_with_spec_non_default(seed):
    from dask.dataframe.io.demo import ColumnSpec, DatasetSpec, RangeIndexSpec, with_spec
    spec = DatasetSpec(npartitions=3, nrecords=10, index_spec=RangeIndexSpec(dtype='int32', step=2), column_specs=[ColumnSpec(prefix='i', dtype='int32', low=1, high=100, random=True), ColumnSpec(prefix='f', dtype='float32', random=True), ColumnSpec(prefix='c', dtype='category', choices=['apple', 'banana']), ColumnSpec(prefix='s', dtype=str, length=15, random=True)])
    ddf = with_spec(spec, seed=seed)
    assert isinstance(ddf, dd.DataFrame)
    assert ddf.columns.tolist() == ['i1', 'f1', 'c1', 's1']
    if PANDAS_GE_200:
        assert ddf.index.dtype == 'int32'
    assert ddf['i1'].dtype == 'int32'
    assert ddf['f1'].dtype == 'float32'
    assert ddf['c1'].dtype.name == 'category'
    assert ddf['s1'].dtype == get_string_dtype() if not dd._dask_expr_enabled() else object
    res = ddf.compute().sort_index()
    assert len(res) == 10
    assert set(res.c1.cat.categories) == {'apple', 'banana'}
    assert res.i1.min() >= 1
    assert res.i1.max() <= 100
    assert all((len(s) == 15 for s in res.s1.tolist()))
    assert len(res.s1.unique()) <= 10