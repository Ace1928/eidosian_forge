from __future__ import annotations
import os
from numpy import nan
import numpy as np
import pandas as pd
import xarray as xr
import datashader as ds
import pytest
from datashader.datatypes import RaggedDtype
def test_canvas_size():
    cvs_list = [ds.Canvas(plot_width=0, plot_height=6), ds.Canvas(plot_width=5, plot_height=0), ds.Canvas(plot_width=0, plot_height=0), ds.Canvas(plot_width=-1, plot_height=1), ds.Canvas(plot_width=10, plot_height=-1)]
    msg = 'Invalid size: plot_width and plot_height must be bigger than 0'
    df = pd.DataFrame(dict(x=[0, 0.2, 1], y=[0, 0.4, 1], z=[10, 20, 30]))
    for cvs in cvs_list:
        with pytest.raises(ValueError, match=msg):
            cvs.points(df, 'x', 'y', ds.mean('z'))