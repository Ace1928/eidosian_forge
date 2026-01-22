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
@open_rasterio_available
def test_out_of_bounds_return_correct_size(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        cvs = ds.Canvas(plot_width=2, plot_height=2, x_range=[10000000000.0, 1e+20], y_range=[10000000000.0, 1e+20])
        try:
            cvs.raster(src)
        except ValueError:
            pass
        else:
            assert False