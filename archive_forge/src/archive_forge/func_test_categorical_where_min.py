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
def test_categorical_where_min(df):
    sol_rowindex = xr.DataArray([[[0, 1, -1, 3], [12, 13, 10, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]], coords=coords + [['a', 'b', 'c', 'd']], dims=dims + ['cat2'])
    sol_reverse = xr.where(np.logical_or(sol_rowindex < 0, sol_rowindex == 6), np.nan, 20 - sol_rowindex)
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'))))
    assert_eq_xr(agg, sol_rowindex)
    agg = c.points(df, 'x', 'y', ds.by('cat2', ds.where(ds.min('plusminus'), 'reverse')))
    assert_eq_xr(agg, sol_reverse)