from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
def test_eq_hist():
    data = np.random.normal(size=300 ** 2)
    data[np.random.randint(300 ** 2, size=100)] = np.nan
    data = (data - np.nanmin(data)).reshape((300, 300))
    mask = np.isnan(data)
    eq, _ = tf.eq_hist(data, mask)
    check_eq_hist_cdf_slope(eq)
    assert (np.isnan(eq) == mask).all()
    data = np.random.normal(scale=100, size=(300, 300)).astype('i8')
    data = data - data.min()
    eq, _ = tf.eq_hist(data)
    check_eq_hist_cdf_slope(eq)