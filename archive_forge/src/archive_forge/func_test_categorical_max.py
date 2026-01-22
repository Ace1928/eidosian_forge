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
def test_categorical_max(df):
    sol_int = np.array([[[4, 1, 2, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]], dtype=np.float64)
    sol_float = np.array([[[4, 1, nan, 3], [12, 13, 14, 11]], [[8, 9, 6, 7], [16, 17, 18, 19]]])
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('i32'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('i64'))).data, sol_int)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('f32'))).data, sol_float)
    assert_eq_ndarray(c.points(df, 'x', 'y', ds.by('cat2', ds.max('f64'))).data, sol_float)