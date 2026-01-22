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
def test_max_row_index(df):
    out = xr.DataArray([[4, 14], [9, 19]], coords=coords, dims=dims)
    agg = c.points(df, 'x', 'y', ds._max_row_index())
    assert agg.dtype == np.int64
    assert_eq_xr(agg, out)