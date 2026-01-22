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
def test_raster_x_ascending_y_descending_partial_range():
    """
    Assert raster with ascending x- and descending y-coordinates is aggregated correctly.
    """
    xs = np.arange(10)
    ys = np.arange(5)[::-1]
    arr = xs * ys[np.newaxis].T
    xarr = xr.DataArray(arr, coords={'X': xs, 'Y': ys}, dims=['Y', 'X'])
    cvs = ds.Canvas(7, 2, x_range=(0.5, 7.5), y_range=(1.5, 3.5))
    agg = cvs.raster(xarr)
    assert np.allclose(agg.data, xarr.sel(X=slice(1, 7), Y=slice(3, 2)).data)
    assert np.allclose(agg.X.values, xs[1:8])
    assert np.allclose(agg.Y.values, ys[1:3])
    assert np.allclose(agg.x_range, (0.5, 7.5))
    assert np.allclose(agg.y_range, (1.5, 3.5))