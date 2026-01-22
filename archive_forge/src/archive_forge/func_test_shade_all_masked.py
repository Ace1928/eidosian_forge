from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
@pytest.mark.parametrize('empty_array', empty_arrays)
def test_shade_all_masked(empty_array):
    agg = xr.DataArray(data=empty_array, coords=dict(y=[0, 1], x=[0, 1], cat=['a', 'b']))
    im = tf.shade(agg, how='eq_hist', cmap=['white', 'white'])
    assert isinstance(im.data, np.ndarray)
    assert im.shape == (2, 2)