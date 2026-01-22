from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_line_antialias_categorical():
    df = pd.DataFrame(dict(x=np.asarray([0, 1, 1, 0, np.nan, 0, 1 / 3.0, 2 / 3.0, 1]), y=np.asarray([0, 1, 0, 1, np.nan, 0.125, 0.15, 0.175, 0.2]), cat=[1, 1, 1, 1, 1, 2, 2, 2, 2]))
    df['cat'] = df['cat'].astype('category')
    x_range = y_range = (-0.1875, 1.1875)
    cvs = ds.Canvas(plot_width=11, plot_height=11, x_range=x_range, y_range=y_range)
    agg = cvs.line(source=df, x='x', y='y', line_width=1, agg=ds.by('cat', ds.count(self_intersect=False)))
    assert_eq_ndarray(agg.data[:, :, 0], line_antialias_sol_0, close=True)
    assert_eq_ndarray(agg.data[:, :, 1], line_antialias_sol_1, close=True)
    agg = cvs.line(source=df, x='x', y='y', line_width=1, agg=ds.by('cat', ds.count(self_intersect=True)))
    assert_eq_ndarray(agg.data[:, :, 0], line_antialias_sol_0_intersect, close=True)
    assert_eq_ndarray(agg.data[:, :, 1], line_antialias_sol_1, close=True)