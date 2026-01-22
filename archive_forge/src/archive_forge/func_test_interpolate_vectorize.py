from __future__ import annotations
from itertools import combinations, permutations
from typing import cast
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding.cftimeindex import _parse_array_of_cftime_strings
from xarray.core.types import InterpOptions
from xarray.tests import (
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize('use_dask', [False, True])
def test_interpolate_vectorize(use_dask: bool) -> None:
    if not has_scipy:
        pytest.skip('scipy is not installed.')
    if not has_dask and use_dask:
        pytest.skip('dask is not installed in the environment.')

    def func(obj, dim, new_x):
        shape = [s for i, s in enumerate(obj.shape) if i != obj.get_axis_num(dim)]
        for s in new_x.shape[::-1]:
            shape.insert(obj.get_axis_num(dim), s)
        return scipy.interpolate.interp1d(da[dim], obj.data, axis=obj.get_axis_num(dim), bounds_error=False, fill_value=np.nan)(new_x).reshape(shape)
    da = get_example_data(0)
    if use_dask:
        da = da.chunk({'y': 5})
    xdest = xr.DataArray(np.linspace(0.1, 0.9, 30), dims='z', coords={'z': np.random.randn(30), 'z2': ('z', np.random.randn(30))})
    actual = da.interp(x=xdest, method='linear')
    expected = xr.DataArray(func(da, 'x', xdest), dims=['z', 'y'], coords={'z': xdest['z'], 'z2': xdest['z2'], 'y': da['y'], 'x': ('z', xdest.values), 'x2': ('z', func(da['x2'], 'x', xdest))})
    assert_allclose(actual, expected.transpose('z', 'y', transpose_coords=True))
    xdest = xr.DataArray(np.linspace(0.1, 0.9, 30).reshape(6, 5), dims=['z', 'w'], coords={'z': np.random.randn(6), 'w': np.random.randn(5), 'z2': ('z', np.random.randn(6))})
    actual = da.interp(x=xdest, method='linear')
    expected = xr.DataArray(func(da, 'x', xdest), dims=['z', 'w', 'y'], coords={'z': xdest['z'], 'w': xdest['w'], 'z2': xdest['z2'], 'y': da['y'], 'x': (('z', 'w'), xdest.data), 'x2': (('z', 'w'), func(da['x2'], 'x', xdest))})
    assert_allclose(actual, expected.transpose('z', 'w', 'y', transpose_coords=True))