from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
@pytest.mark.parametrize('array_module', array_modules)
def test_raster_subpixel_quads_represented(array_module):
    c = ds.Canvas(plot_width=8, plot_height=4, x_range=[0.5, 16.5], y_range=[0.5, 8.5])
    da = xr.DataArray(array_module.array([[1, 2, 3, 4], [5, 6, 7, 8]]), coords=[('b', [1, 2]), ('a', [1, 2, 3, 4])], name='Z')
    y_coords = np.linspace(1.5, 7.5, 4)
    x_coords = np.linspace(1.5, 15.5, 8)
    out = xr.DataArray(array_module.array([[14.0, 22.0, nan, nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan], [nan, nan, nan, nan, nan, nan, nan, nan]], dtype='f4'), coords=[('b', y_coords), ('a', x_coords)])
    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)