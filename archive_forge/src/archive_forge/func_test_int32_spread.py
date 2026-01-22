from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_int32_spread():
    data = np.array([[1, 1, 0, 0, 0], [1, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 3, 0], [0, 0, 0, 0, 0]], dtype='int32')
    coords = [np.arange(5), np.arange(5)]
    arr = xr.DataArray(data, coords=coords, dims=dims)
    s = tf.spread(arr)
    o = np.array([[5, 5, 3, 0, 0], [5, 5, 3, 0, 0], [3, 3, 5, 3, 3], [0, 0, 3, 3, 3], [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)
    assert (s.x_axis == arr.x_axis).all()
    assert (s.y_axis == arr.y_axis).all()
    assert s.dims == arr.dims
    s = tf.spread(arr, px=2)
    o = np.array([[5, 5, 5, 3, 0], [5, 5, 8, 6, 3], [5, 8, 7, 5, 3], [3, 6, 5, 3, 3], [0, 3, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)
    s = tf.spread(arr, shape='square')
    o = np.array([[5, 5, 3, 0, 0], [5, 5, 3, 0, 0], [3, 3, 5, 3, 3], [0, 0, 3, 3, 3], [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)
    s = tf.spread(arr, how='min')
    o = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.data, o)
    s = tf.spread(arr, how='max')
    o = np.array([[2, 2, 2, 0, 0], [2, 2, 2, 0, 0], [2, 2, 3, 3, 3], [0, 0, 3, 3, 3], [0, 0, 3, 3, 3]])
    np.testing.assert_equal(s.data, o)
    mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    data = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='int32')
    arr = xr.DataArray(data, coords=coords, dims=dims)
    s = tf.spread(arr, mask=mask)
    o = np.array([[0, 0, 0, 1, 0], [1, 0, 2, 0, 1], [0, 1, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
    np.testing.assert_equal(s.data, o)
    s = tf.spread(arr, px=0)
    np.testing.assert_equal(s.data, arr.data)
    pytest.raises(ValueError, lambda: tf.spread(arr, px=-1))
    pytest.raises(ValueError, lambda: tf.spread(arr, mask=np.ones(2)))
    pytest.raises(ValueError, lambda: tf.spread(arr, mask=np.ones((2, 2))))