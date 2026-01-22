from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_array_dynspread():
    coords = [np.arange(5), np.arange(5)]
    data = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype='uint32')
    arr = xr.DataArray(data, coords=coords, dims=dims)
    assert tf.dynspread(arr).equals(arr)
    data = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='uint32')
    arr = xr.DataArray(data, coords=coords, dims=dims)
    assert tf.dynspread(arr, threshold=0.4).equals(tf.spread(arr, 0))
    assert tf.dynspread(arr, threshold=0.7).equals(tf.spread(arr, 1))
    assert tf.dynspread(arr, threshold=1.0).equals(tf.spread(arr, 3))
    assert tf.dynspread(arr, max_px=0).equals(arr)
    pytest.raises(ValueError, lambda: tf.dynspread(arr, threshold=1.1))
    pytest.raises(ValueError, lambda: tf.dynspread(arr, max_px=-1))