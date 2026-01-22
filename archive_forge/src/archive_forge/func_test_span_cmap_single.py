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
@pytest.mark.parametrize('cmap', ['black', (0, 0, 0), '#000000'])
def test_span_cmap_single(agg, cmap):
    x = agg.a
    sol = np.array([[0, 671088640, 1946157056], [2701131776, 0, 3640655872], [3976200192, 4278190080, 0]])
    sol = tf.Image(sol, coords=coords, dims=dims)
    check_span(x, cmap, 'log', sol)