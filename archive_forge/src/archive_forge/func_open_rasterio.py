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
def open_rasterio(path, *args, **kwargs):
    if Version(xr.__version__) < Version('0.20'):
        func = xr.open_rasterio
    else:
        func = rioxarray.open_rasterio
    return func(path, *args, **kwargs)