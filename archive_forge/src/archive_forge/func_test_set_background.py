from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_set_background():
    out = tf.set_background(img1)
    assert out.equals(img1)
    sol = tf.Image(np.array([[4278255615, 4278190335], [4278190335, 4278255485]], dtype='uint32'), coords=coords2, dims=dims)
    out = tf.set_background(img1, 'red')
    assert out.equals(sol)