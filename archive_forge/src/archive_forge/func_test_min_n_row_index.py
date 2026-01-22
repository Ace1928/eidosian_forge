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
def test_min_n_row_index(df):
    solution = np.array([[[0, 1, 2, 3, 4, -1], [10, 11, 12, 13, 14, -1]], [[5, 6, 7, 8, 9, -1], [15, 16, 17, 18, 19, -1]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds._min_n_row_index(n=n))
        assert agg.dtype == np.int64
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds._min_row_index()).data)