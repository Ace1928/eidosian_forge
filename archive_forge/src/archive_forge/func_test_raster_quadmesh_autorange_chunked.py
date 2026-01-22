from __future__ import annotations
import numpy as np
from numpy import nan
import xarray as xr
import datashader as ds
import pytest
import dask.array
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
def test_raster_quadmesh_autorange_chunked():
    c = ds.Canvas(plot_width=8, plot_height=6)
    da = xr.DataArray(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), coords=[('b', [1, 2, 3]), ('a', [1, 2, 3, 4])], name='Z').chunk({'a': 2, 'b': 2})
    y_coords = np.linspace(0.75, 3.25, 6)
    x_coords = np.linspace(0.75, 4.25, 8)
    out = xr.DataArray(np.array([[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0], [5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0], [5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0], [9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0], [9.0, 9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0]], dtype='f8'), coords=[('b', y_coords), ('a', x_coords)])
    res = c.quadmesh(da, x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)
    res = c.quadmesh(da.transpose('a', 'b'), x='a', y='b', agg=ds.sum('Z'))
    assert_eq_xr(res, out)