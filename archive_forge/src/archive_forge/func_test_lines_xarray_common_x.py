from __future__ import annotations
import numpy as np
from numpy import nan
import os
import xarray as xr
import datashader as ds
from datashader.tests.test_pandas import assert_eq_ndarray
import pytest
@pytest.mark.parametrize('ds2d', ds2ds)
@pytest.mark.parametrize('cuda', [False, True])
@pytest.mark.parametrize('chunksizes', [None, dict(x=10, channel=10), dict(x=10, channel=1), dict(x=3, channel=10), dict(x=3, channel=1)])
def test_lines_xarray_common_x(ds2d, cuda, chunksizes):
    source = ds2d.copy()
    if cuda:
        if not (cupy and test_gpu):
            pytest.skip('CUDA tests not requested')
        elif chunksizes is not None:
            pytest.skip('CUDA-dask for LinesXarrayCommonX not implemented')
        source.name.data = cupy.asarray(source.name.data)
    if chunksizes is not None:
        source = source.chunk(chunksizes)
    canvas = ds.Canvas(plot_height=3, plot_width=7)
    sol_count = np.array([[0, 0, 1, 1, 0, 0, 0], [1, 2, 1, 1, 2, 2, 1], [1, 0, 0, 0, 0, 0, 1]], dtype=np.uint32)
    sol_max = np.array([[nan, nan, -33, -33, nan, nan, nan], [-55, -33, -55, -55, -33, -33, -55], [-33, nan, nan, nan, nan, nan, -33]], dtype=np.float64)
    sol_min = np.array([[nan, nan, -33, -33, nan, nan, nan], [-55, -55, -55, -55, -55, -55, -55], [-33, nan, nan, nan, nan, nan, -33]], dtype=np.float64)
    sol_sum = np.array([[nan, nan, -33, -33, nan, nan, nan], [-55, -88, -55, -55, -88, -88, -55], [-33, nan, nan, nan, nan, nan, -33]], dtype=np.float64)
    sol_max_row_index = np.array([[-1, -1, 0, 0, -1, -1, -1], [1, 1, 1, 1, 1, 1, 1], [0, -1, -1, -1, -1, -1, 0]], dtype=np.int64)
    sol_min_row_index = np.array([[-1, -1, 0, 0, -1, -1, -1], [1, 0, 1, 1, 0, 0, 1], [0, -1, -1, -1, -1, -1, 0]], dtype=np.int64)
    if chunksizes is not None and chunksizes['x'] == 3:
        sol_count[:, 4] = 0
        sol_max[:, 4] = nan
        sol_min[:, 4] = nan
        sol_sum[:, 4] = nan
        sol_max_row_index[:, 4] = -1
        sol_min_row_index[:, 4] = -1
    sol_first = np.select([sol_min_row_index == 0, sol_min_row_index == 1], value, np.nan)
    sol_last = np.select([sol_max_row_index == 0, sol_max_row_index == 1], value, np.nan)
    sol_where_max_other = np.select([sol_max == -33, sol_max == -55], other, np.nan)
    sol_where_max_row = np.select([sol_max == -33, sol_max == -55], [0, 1], -1)
    sol_where_min_other = np.select([sol_min == -33, sol_min == -55], other, np.nan)
    sol_where_min_row = np.select([sol_min == -33, sol_min == -55], [0, 1], -1)
    agg = canvas.line(source, x='x', y='name', agg=ds.count())
    assert_eq_ndarray(agg.x_range, (0, 4), close=True)
    assert_eq_ndarray(agg.y_range, (0, 2), close=True)
    assert_eq_ndarray(agg.data, sol_count)
    assert isinstance(agg.data, cupy.ndarray if cuda else np.ndarray)
    agg = canvas.line(source, x='x', y='name', agg=ds.any())
    assert_eq_ndarray(agg.data, sol_count > 0)
    agg = canvas.line(source, x='x', y='name', agg=ds.max('value'))
    assert_eq_ndarray(agg.data, sol_max)
    agg = canvas.line(source, x='x', y='name', agg=ds.min('value'))
    assert_eq_ndarray(agg.data, sol_min)
    agg = canvas.line(source, x='x', y='name', agg=ds.sum('value'))
    assert_eq_ndarray(agg.data, sol_sum)
    agg = canvas.line(source, x='x', y='name', agg=ds._max_row_index())
    assert_eq_ndarray(agg.data, sol_max_row_index)
    agg = canvas.line(source, x='x', y='name', agg=ds._min_row_index())
    assert_eq_ndarray(agg.data, sol_min_row_index)
    agg = canvas.line(source, x='x', y='name', agg=ds.first('value'))
    assert_eq_ndarray(agg.data, sol_first)
    agg = canvas.line(source, x='x', y='name', agg=ds.last('value'))
    assert_eq_ndarray(agg.data, sol_last)
    agg = canvas.line(source, x='x', y='name', agg=ds.where(ds.max('value'), 'other'))
    assert_eq_ndarray(agg.data, sol_where_max_other)
    agg = canvas.line(source, x='x', y='name', agg=ds.where(ds.max('value')))
    assert_eq_ndarray(agg.data, sol_where_max_row)
    agg = canvas.line(source, x='x', y='name', agg=ds.where(ds.min('value'), 'other'))
    assert_eq_ndarray(agg.data, sol_where_min_other)
    agg = canvas.line(source, x='x', y='name', agg=ds.where(ds.min('value')))
    assert_eq_ndarray(agg.data, sol_where_min_row)