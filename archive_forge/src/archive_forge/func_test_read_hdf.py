from __future__ import annotations
import os
import pathlib
from time import sleep
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask._compatibility import PY_VERSION
from dask.dataframe._compat import tm
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import assert_eq
from dask.layers import DataFrameIOLayer
from dask.utils import dependency_depth, tmpdir, tmpfile
@pytest.mark.parametrize('data, compare', [(pd.DataFrame({'x': ['a', 'b', 'c', 'd'], 'y': [1, 2, 3, 4]}, index=[1.0, 2.0, 3.0, 4.0]), tm.assert_frame_equal), (pd.Series([1, 2, 3, 4], name='a'), tm.assert_series_equal)])
def test_read_hdf(data, compare):
    pytest.importorskip('tables')
    with tmpfile('h5') as fn:
        data.to_hdf(fn, key='/data')
        try:
            dd.read_hdf(fn, 'data', chunksize=2, mode='r')
            assert False
        except TypeError as e:
            assert "format='table'" in str(e)
    with tmpfile('h5') as fn:
        data.to_hdf(fn, key='/data', format='table')
        a = dd.read_hdf(fn, '/data', chunksize=2, mode='r')
        assert a.npartitions == 2
        compare(a.compute(), data)
        compare(dd.read_hdf(fn, '/data', chunksize=2, start=1, stop=3, mode='r').compute(), pd.read_hdf(fn, '/data', start=1, stop=3))
        assert sorted(dd.read_hdf(fn, '/data', mode='r').dask) == sorted(dd.read_hdf(fn, '/data', mode='r').dask)
    with tmpfile('h5') as fn:
        sorted_data = data.sort_index()
        sorted_data.to_hdf(fn, key='/data', format='table')
        a = dd.read_hdf(fn, '/data', chunksize=2, sorted_index=True, mode='r')
        assert a.npartitions == 2
        compare(a.compute(), sorted_data)