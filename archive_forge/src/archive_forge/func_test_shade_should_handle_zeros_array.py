from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_shade_should_handle_zeros_array():
    data = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype='uint32')
    arr = tf.Image(data, dims=['x', 'y'])
    img = tf.shade(arr, cmap=['white', 'black'], how='linear')
    assert img is not None