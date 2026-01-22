from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_make_timeseries_column_projection():
    ddf = dd.demo.make_timeseries('2001', '2002', freq='1D', partition_freq=f'3{ME}', seed=42)
    assert_eq(ddf[['x']].compute(), ddf.compute()[['x']])
    assert_eq(ddf.groupby('name').aggregate({'x': 'sum', 'y': 'max'}).compute(), ddf.compute().groupby('name').aggregate({'x': 'sum', 'y': 'max'}))