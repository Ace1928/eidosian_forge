import pytest
import pandas as pd
import numpy as np
import xarray as xr
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray, assert_eq_xr
import dask.dataframe as dd
@pytest.mark.skipif(not spatialpandas, reason='spatialpandas not installed')
@pytest.mark.parametrize('DataFrame', DataFrames)
def test_no_overlap_agg(DataFrame):
    df = DataFrame({'polygons': pd.Series([[[1, 1, 2, 2, 1, 3, 0, 2, 1, 1], [0.5, 1.5, 0.5, 2.5, 1.5, 2.5, 1.5, 1.5, 0.5, 1.5]], [[0.5, 1.5, 1.5, 1.5, 1.5, 2.5, 0.5, 2.5, 0.5, 1.5]], [[0, 1, 2, 1, 2, 3, 0, 3, 0, 1, 1, 1, 0, 2, 1, 3, 2, 2, 1, 1]]], dtype='Polygon[float64]'), 'v': range(3)})
    cvs = ds.Canvas(plot_width=16, plot_height=16)
    agg = cvs.polygons(df, geometry='polygons', agg=ds.sum('v'))
    axis = ds.core.LinearAxis()
    lincoords_x = axis.compute_index(axis.compute_scale_and_translate((0, 2), 16), 16)
    lincoords_y = axis.compute_index(axis.compute_scale_and_translate((1, 3), 16), 16)
    sol = np.array([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0], [2.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    out = xr.DataArray(sol, coords=[lincoords_y, lincoords_x], dims=['y', 'x'])
    assert_eq_xr(agg, out)