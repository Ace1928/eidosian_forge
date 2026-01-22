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
def test_raster_float_nan_value_padding():
    """
    Ensure that the padding values respect the supplied nan_value.
    """
    cvs = ds.Canvas(plot_height=3, plot_width=3, x_range=(0, 2), y_range=(0, 2))
    array = np.array([[np.nan, 1.0, 2.0, 3.0], [4.0, np.nan, 6.0, 7.0], [8.0, 9.0, np.nan, 11.0]])
    xr_array = xr.DataArray(array, coords={'x': np.linspace(0, 1, 4), 'y': np.linspace(0, 1, 3)}, dims=['y', 'x'])
    agg = cvs.raster(xr_array, downsample_method='max')
    expected = np.array([[4.0, 7.0, np.nan], [9.0, 11.0, np.nan], [np.nan, np.nan, np.nan]])
    assert np.allclose(agg.data, expected, equal_nan=True)
    assert agg.data.dtype.kind == 'f'
    assert np.allclose(agg.x.values, np.array([1 / 3.0, 1.0, 5 / 3.0]))
    assert np.allclose(agg.y.values, np.array([1 / 3.0, 1.0, 5 / 3.0]))