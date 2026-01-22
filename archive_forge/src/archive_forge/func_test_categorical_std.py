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
def test_categorical_std(df):
    sol = np.sqrt(np.array([[[2.5, nan, nan, nan], [nan, nan, 2.0, nan]], [[nan, 2.0, nan, nan], [nan, nan, nan, 2.0]]]))
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat'])
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.std('f32')))
    assert_eq_xr(agg, out, True)
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.std('f64')))
    assert_eq_xr(agg, out, True)
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=dims + ['cat_int'])
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.std('f32')))
    assert_eq_xr(agg, out)
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.std('f64')))
    assert_eq_xr(agg, out)
    sol = np.append(sol, [[[nan], [nan]], [[nan], [nan]]], axis=2)
    for col in ('f32', 'f64'):
        out = xr.DataArray(sol, coords=coords + [range(5)], dims=dims + [col])
        agg = c.points(df, 'x', 'y', ds.by(ds.category_binning(col, 0, 20, 4), ds.std(col)))
        assert_eq_xr(agg, out)