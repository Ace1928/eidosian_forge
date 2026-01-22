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
@pytest.mark.parametrize('attr', ['a', 'b', 'c'])
@pytest.mark.parametrize('span', [None, int_span, float_span])
def test_shade(agg, attr, span):
    x = getattr(agg, attr)
    cmap = ['pink', 'red']
    img = tf.shade(x, cmap=cmap, how='log', span=span)
    sol = solutions['log']
    assert_eq_xr(img, sol)
    assert list(img.coords) == ['x_axis', 'y_axis']
    assert list(img.dims) == ['y_axis', 'x_axis']
    img = tf.shade(x, cmap=cmap, how='cbrt', span=span)
    sol = solutions['cbrt']
    assert_eq_xr(img, sol)
    img = tf.shade(x, cmap=cmap, how='linear', span=span)
    sol = solutions['linear']
    assert_eq_xr(img, sol)
    if span is None:
        img = tf.shade(x, cmap=cmap, how='eq_hist', rescale_discrete_levels=False)
        sol = tf.Image(eq_hist_sol, coords=coords, dims=dims)
        assert_eq_xr(img, sol)
        img = tf.shade(x, cmap=cmap, how='eq_hist', rescale_discrete_levels=True)
        sol = tf.Image(eq_hist_sol_rescale_discrete_levels[attr], coords=coords, dims=dims)
        assert_image_close(img, sol, tolerance=1)
    img = tf.shade(x, cmap=cmap, how=lambda x, mask: np.where(mask, np.nan, x ** 2))
    sol = np.array([[0, 4291543295, 4291148543], [4290030335, 0, 4285557503], [4282268415, 4278190335, 0]], dtype='u4')
    sol = tf.Image(sol, coords=coords, dims=dims)
    assert_eq_xr(img, sol)