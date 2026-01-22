from __future__ import annotations
import pytest
from dask.context import config
from os import path
from itertools import product
import datashader as ds
import xarray as xr
import numpy as np
import dask.array as da
import pandas as pd
from datashader.resampling import compute_chunksize
import datashader.transfer_functions as tf
from packaging.version import Version
def test_raster_single_pixel_range():
    """
    Ensure that canvas range covering a single pixel are handled correctly.
    """
    cvs = ds.Canvas(plot_height=3, plot_width=3, x_range=(0, 0.1), y_range=(0, 0.1))
    array = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    xr_array = xr.DataArray(array, dims=['y', 'x'], coords={'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)})
    agg = cvs.raster(xr_array, downsample_method='max', nan_value=9999)
    expected = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.allclose(agg.data, expected)
    assert agg.data.dtype.kind == 'i'
    assert np.allclose(agg.x.values, np.array([1 / 60.0, 1 / 20.0, 1 / 12.0]))
    assert np.allclose(agg.y.values, np.array([1 / 60.0, 1 / 20.0, 1 / 12.0]))