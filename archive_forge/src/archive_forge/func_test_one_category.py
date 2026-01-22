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
def test_one_category(df):
    assert len(df['onecat'].unique()) == 1
    sol = np.array([[[5], [5]], [[5], [5]]])
    out = xr.DataArray(sol, coords=coords + [['one']], dims=dims + ['onecat'])
    agg = c.points(df, 'x', 'y', ds.by('onecat', ds.count('i32')))
    assert agg.shape == (2, 2, 1)
    assert_eq_xr(agg, out)