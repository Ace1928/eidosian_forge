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
def test_categorical_max2(df):
    sol = np.array([[[4, nan, nan, nan], [nan, nan, 14, nan]], [[nan, 9, nan, nan], [nan, nan, nan, 19]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat'])
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.max('i32')))
    assert_eq_xr(agg, out)
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=dims + ['cat_int'])
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.max('i32')))
    assert_eq_xr(agg, out)
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.max('i64')))
    assert_eq_xr(agg, out)