from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_area_to_line_autorange_gap():
    axis = ds.core.LinearAxis()
    lincoords_y = axis.compute_index(axis.compute_scale_and_translate((-4.0, 4.0), 7), 7)
    lincoords_x = axis.compute_index(axis.compute_scale_and_translate((-4.0, 4.0), 13), 13)
    cvs = ds.Canvas(plot_width=13, plot_height=7)
    df = pd.DataFrame({'x': [-4, -2, 0, np.nan, 2, 4], 'y0': [0, -4, 0, np.nan, 4, 0], 'y1': [0, 0, 0, np.nan, 0, 0]})
    agg = cvs.area(df, 'x', 'y0', ds.count(), y_stack='y1')
    sol = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype='i4')
    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y0', 'x'])
    assert_eq_xr(agg, out)