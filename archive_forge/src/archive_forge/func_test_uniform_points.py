from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_uniform_points():
    n = 101
    df = pd.DataFrame({'time': np.ones(2 * n, dtype='i4'), 'x': np.concatenate((np.arange(n, dtype='f8'), np.arange(n, dtype='f8'))), 'y': np.concatenate(([0.0] * n, [1.0] * n))})
    cvs = ds.Canvas(plot_width=10, plot_height=2, y_range=(0, 1))
    agg = cvs.points(df, 'x', 'y', ds.count('time'))
    sol = np.array([[10] * 9 + [11], [10] * 9 + [11]], dtype='i4')
    assert_eq_ndarray(agg.data, sol)
    assert_eq_ndarray(agg.x_range, (0, 100), close=True)
    assert_eq_ndarray(agg.y_range, (0, 1), close=True)