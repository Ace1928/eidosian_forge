from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
@pytest.mark.parametrize('array_module', array_modules)
def test_rect_quadmesh_autorange_reversed(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4)
    da = xr.DataArray(array_module.array([[1, 2, 3, 4], [5, 6, 7, 8]]), coords=[('b', [-1, -2]), ('a', [-1, -2, -3, -8])], name='Z')
    y_coords = np.linspace(-2.25, -0.75, 4)
    x_coords = np.linspace(-9.875, -1.125, 8)
    out = xr.DataArray(array_module.array([[8.0, 8.0, 8.0, 8.0, 7.0, 7.0, 6.0, 5.0], [8.0, 8.0, 8.0, 8.0, 7.0, 7.0, 6.0, 5.0], [4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 2.0, 1.0], [4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 2.0, 1.0]], dtype='f8'), coords=[('b', y_coords), ('a', x_coords)])
    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (-10.5, -0.5), close=True)
    assert_eq_ndarray(res.y_range, (-2.5, -0.5), close=True)
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (-10.5, -0.5), close=True)
    assert_eq_ndarray(res.y_range, (-2.5, -0.5), close=True)