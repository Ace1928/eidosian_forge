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
def test_categorical_first_n(df):
    solution = np.array([[[[0, 4, nan], [-1, nan, nan], [nan, nan, nan], [-3, nan, nan]], [[12, nan, nan], [-13, nan, nan], [10, 14, nan], [-11, nan, nan]]], [[[8, nan, nan], [-5, -9, nan], [6, nan, nan], [-7, nan, nan]], [[16, nan, nan], [-17, nan, nan], [18, nan, nan], [-15, -19, nan]]]])
    for n in range(1, 3):
        agg = c.points(df, 'x', 'y', ds.by('cat2', ds.first_n('plusminus', n=n)))
        out = solution[:, :, :, :n]
        assert_eq_ndarray(agg.data, out)
        if n == 1:
            assert_eq_ndarray(agg[..., 0].data, c.points(df, 'x', 'y', ds.by('cat2', ds.first('plusminus'))).data)