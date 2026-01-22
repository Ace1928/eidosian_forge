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
def test_categorical_sum_binning(df):
    sol = np.array([[[8.0, nan, nan, nan], [nan, nan, 60.0, nan]], [[nan, 35.0, nan, nan], [nan, nan, nan, 85.0]]])
    sol = np.append(sol, [[[nan], [nan]], [[nan], [nan]]], axis=2)
    for col in ('f32', 'f64'):
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=dims + [col])
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.sum(col)))
        assert_eq_xr(agg, out)
        assert_eq_ndarray(agg.x_range, (0, 1), close=True)
        assert_eq_ndarray(agg.y_range, (0, 1), close=True)