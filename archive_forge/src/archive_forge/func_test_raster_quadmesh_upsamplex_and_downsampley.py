from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
@pytest.mark.parametrize('array_module', array_modules)
def test_raster_quadmesh_upsamplex_and_downsampley(array_module):
    c = ds.Canvas(plot_width=4, plot_height=2)
    da = xr.DataArray(array_module.array([[1, 2], [3, 4], [5, 6], [7, 8]]), coords=[('b', [1, 2, 3, 4]), ('a', [1, 2])], name='Z')
    x_coords = np.linspace(0.75, 2.25, 4)
    y_coords = np.linspace(1.5, 3.5, 2)
    out = xr.DataArray(array_module.array([[4.0, 4.0, 6.0, 6.0], [12.0, 12.0, 14.0, 14.0]], dtype='f8'), coords=[('b', y_coords), ('a', x_coords)])
    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)