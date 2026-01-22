from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_points_on_edge():
    df = pd.DataFrame(dict(x=[0, 0.5, 1.1, 1.5, 2.2, 3, 3, 0], y=[0, 0, 0, 0, 0, 0, 3, 3]))
    canvas = ds.Canvas(plot_width=3, plot_height=3, x_range=(0, 3), y_range=(0, 3))
    agg = canvas.points(df, 'x', 'y', agg=ds.count())
    sol = np.array([[2, 2, 2], [0, 0, 0], [1, 0, 1]], dtype='int32')
    out = xr.DataArray(sol, coords=[('x', [0.5, 1.5, 2.5]), ('y', [0.5, 1.5, 2.5])], dims=['y', 'x'])
    assert_eq_xr(agg, out)