from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_with_spec_datetime_index():
    from dask.dataframe.io.demo import ColumnSpec, DatasetSpec, DatetimeIndexSpec, with_spec
    spec = DatasetSpec(nrecords=10, index_spec=DatetimeIndexSpec(dtype='datetime64[ns]', freq='1h', start='2023-01-02', partition_freq='1D'), column_specs=[ColumnSpec(dtype=int)])
    ddf = with_spec(spec, seed=42)
    assert ddf.index.dtype == 'datetime64[ns]'
    res = ddf.compute()
    assert len(res) == 10