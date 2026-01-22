from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_with_spec_category_nunique():
    from dask.dataframe.io.demo import ColumnSpec, DatasetSpec, with_spec
    spec = DatasetSpec(npartitions=1, nrecords=20, column_specs=[ColumnSpec(dtype='category', nunique=10)])
    ddf = with_spec(spec, seed=42)
    res = ddf.compute()
    assert res.category1.cat.categories.tolist() == ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']