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
def test_min_n(df):
    solution = np.array([[[-3, -1, 0, 4, nan, nan], [-13, -11, 10, 12, 14, nan]], [[-9, -7, -5, 6, 8, nan], [-19, -17, -15, 16, 18, nan]]])
    for n in range(1, 7):
        agg = c.points(df, 'x', 'y', ds.min_n('plusminus', n=n))
        out = solution[:, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[:, :, 0].data, c.points(df, 'x', 'y', ds.min('plusminus')).data)