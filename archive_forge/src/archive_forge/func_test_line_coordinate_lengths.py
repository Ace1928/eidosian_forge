from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_line_coordinate_lengths():
    cvs = ds.Canvas(plot_width=10, plot_height=6)
    msg = '^x and y coordinate lengths do not match'
    df = pd.DataFrame(dict(x0=[0, 0.2, 1], y0=[0, 0.4, 1], x1=[0, 0.6, 1], y1=[1, 0.8, 1]))
    for axis in (0, 1):
        with pytest.raises(ValueError, match=msg):
            cvs.line(source=df, x=['x0'], y=['y0', 'y1'], axis=axis)
        with pytest.raises(ValueError, match=msg):
            cvs.line(source=df, x=['x0', 'x1'], y=['y0'], axis=axis)
    df = pd.DataFrame(dict(y0=[0, 1, 0, 1], y1=[0, 1, 1, 0]))
    for nx in (1, 3):
        with pytest.raises(ValueError, match=msg):
            cvs.line(source=df, x=np.arange(nx), y=['y0', 'y1'], axis=1)
    df = pd.DataFrame(dict(x0=[0, 1, 0, 1], x1=[0, 1, 1, 0]))
    for ny in (1, 3):
        with pytest.raises(ValueError, match=msg):
            cvs.line(source=df, x=['x0', 'x1'], y=np.arange(ny), axis=1)