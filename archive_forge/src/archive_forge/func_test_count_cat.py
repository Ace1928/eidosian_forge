from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
@pytest.mark.parametrize('df', dfs)
def test_count_cat(df):
    sol = np.array([[[5, 0, 0, 0], [0, 0, 5, 0]], [[0, 5, 0, 0], [0, 0, 0, 5]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat'])
    agg = c.points(df, 'x', 'y', ds.count_cat('cat'))
    assert_eq_xr(agg, out)
    assert_eq_ndarray(agg.x_range, (0, 1), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)