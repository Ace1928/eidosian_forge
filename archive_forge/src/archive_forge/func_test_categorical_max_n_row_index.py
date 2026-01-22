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
def test_categorical_max_n_row_index(df):
    solution = np.array([[[[4, 0, -1], [1, -1, -1], [2, -1, -1], [3, -1, -1]], [[12, -1, -1], [13, -1, -1], [14, 10, -1], [11, -1, -1]]], [[[8, -1, -1], [9, 5, -1], [6, -1, -1], [7, -1, -1]], [[16, -1, -1], [17, -1, -1], [18, -1, -1], [19, 15, -1]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds._max_n_row_index(n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds._max_row_index())).data)