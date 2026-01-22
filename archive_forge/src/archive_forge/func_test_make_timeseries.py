from __future__ import annotations
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.blockwise import Blockwise, optimize_blockwise
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq, get_string_dtype
def test_make_timeseries():
    df = dd.demo.make_timeseries('2000', '2015', {'A': float, 'B': int, 'C': str}, freq='2D', partition_freq=f'6{ME}')
    assert df.divisions[0] == pd.Timestamp('2000-01-31')
    assert df.divisions[-1] == pd.Timestamp('2014-07-31')
    tm.assert_index_equal(df.columns, pd.Index(['A', 'B', 'C']))
    assert df['A'].head().dtype == float
    assert df['B'].head().dtype == int
    assert df['C'].head().dtype == get_string_dtype() if not dd._dask_expr_enabled() else object
    assert df.index.name == 'timestamp'
    assert df.head().index.name == df.index.name
    assert df.divisions == tuple(pd.date_range(start='2000', end='2015', freq=f'6{ME}'))
    tm.assert_frame_equal(df.head(), df.head())
    a = dd.demo.make_timeseries('2000', '2015', {'A': float, 'B': int, 'C': str}, freq='2D', partition_freq=f'6{ME}', seed=123)
    b = dd.demo.make_timeseries('2000', '2015', {'A': float, 'B': int, 'C': str}, freq='2D', partition_freq=f'6{ME}', seed=123)
    c = dd.demo.make_timeseries('2000', '2015', {'A': float, 'B': int, 'C': str}, freq='2D', partition_freq=f'6{ME}', seed=456)
    d = dd.demo.make_timeseries('2000', '2015', {'A': float, 'B': int, 'C': str}, freq='2D', partition_freq=f'3{ME}', seed=123)
    e = dd.demo.make_timeseries('2000', '2015', {'A': float, 'B': int, 'C': str}, freq='1D', partition_freq=f'6{ME}', seed=123)
    tm.assert_frame_equal(a.head(), b.head())
    assert not (a.head(10) == c.head(10)).all().all()
    assert a._name == b._name
    assert a._name != c._name
    assert a._name != d._name
    assert a._name != e._name