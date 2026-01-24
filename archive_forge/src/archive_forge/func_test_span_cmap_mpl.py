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
def test_span_cmap_mpl(agg):
    x = agg.a
    cm = pytest.importorskip('matplotlib.cm')
    cmap = cm.viridis
    sol = np.array([[0, 4283695428, 4287524142], [4287143710, 0, 4282832267], [4280213706, 4280608928, 0]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    check_span(x, cmap, 'log', sol)