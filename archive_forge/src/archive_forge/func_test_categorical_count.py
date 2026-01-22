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
def test_categorical_count(df):
    sol = np.array([[[5, 0, 0, 0], [0, 0, 5, 0]], [[0, 5, 0, 0], [0, 0, 0, 5]]])
    out = xr.DataArray(sol, coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat'])
    agg = c.points(df, 'x', 'y', ds.by('cat', ds.count('i32')))
    assert_eq_xr(agg, out)
    dataset = c.points(df, 'x', 'y', ds.summary(name=ds.by('cat', ds.count('i32'))))
    assert_eq_xr(dataset['name'], out)
    out = xr.DataArray(sol, coords=coords + [range(4)], dims=dims + ['cat_int'])
    agg = c.points(df, 'x', 'y', ds.by(ds.category_modulo('cat_int', modulo=4, offset=10), ds.count()))
    assert_eq_xr(agg, out)