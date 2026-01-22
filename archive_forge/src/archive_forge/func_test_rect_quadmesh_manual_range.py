from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
@pytest.mark.parametrize('array_module', array_modules)
def test_rect_quadmesh_manual_range(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4, x_range=[1, 3], y_range=[-1, 3])
    da = xr.DataArray(array_module.array([[1, 2, 3, 4], [5, 6, 7, 8]]), coords=[('b', [1, 2]), ('a', [1, 2, 3, 8])], name='Z')
    y_coords = np.linspace(-0.5, 2.5, 4)
    x_coords = np.linspace(1.125, 2.875, 8)
    out = xr.DataArray(array_module.array([[nan, nan, nan, nan, nan, nan, nan, nan], [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0], [5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0], [nan, nan, nan, nan, nan, nan, nan, nan]], dtype='f8'), coords=[('b', y_coords), ('a', x_coords)])
    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (1, 3), close=True)
    assert_eq_ndarray(res.y_range, (-1, 3), close=True)
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out, close=True)
    assert_eq_ndarray(res.x_range, (1, 3), close=True)
    assert_eq_ndarray(res.y_range, (-1, 3), close=True)