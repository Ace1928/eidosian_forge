from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def build_agg(array_module=np):
    a = array_module.arange(10, 19, dtype='u4').reshape((3, 3))
    a[[0, 1, 2], [0, 1, 2]] = 0
    s_a = xr.DataArray(a, coords=coords, dims=dims)
    b = array_module.arange(10, 19, dtype='f4').reshape((3, 3))
    b[[0, 1, 2], [0, 1, 2]] = array_module.nan
    s_b = xr.DataArray(b, coords=coords, dims=dims)
    c = array_module.arange(10, 19, dtype='f8').reshape((3, 3))
    c[[0, 1, 2], [0, 1, 2]] = array_module.nan
    s_c = xr.DataArray(c, coords=coords, dims=dims)
    d = array_module.arange(10, 19, dtype='u4').reshape((3, 3))
    d[[0, 1, 2, 2], [0, 1, 2, 1]] = 1
    s_d = xr.DataArray(d, coords=coords, dims=dims)
    agg = xr.Dataset(dict(a=s_a, b=s_b, c=s_c, d=s_d))
    return agg