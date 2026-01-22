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
def test_raster_aggregate_nearest(cvs):
    with open_rasterio(TEST_RASTER_PATH) as src:
        agg = cvs.raster(src, upsample_method='nearest')
        assert agg is not None