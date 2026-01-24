from __future__ import annotations
from io import BytesIO
import numpy as np
import xarray as xr
import dask.array as da
import PIL
import pytest
import datashader.transfer_functions as tf
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr, assert_image_close
@pytest.mark.parametrize('agg', aggs)
def test_shade_mpl_cmap(agg):
    cm = pytest.importorskip('matplotlib.cm')
    img = tf.shade(agg.a, how='log', cmap=cm.viridis)
    sol = np.array([[0, 4283695428, 4287524142], [4287143710, 0, 4282832267], [4280213706, 4280608928, 0]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)