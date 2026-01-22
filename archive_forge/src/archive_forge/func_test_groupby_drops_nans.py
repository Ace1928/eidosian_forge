from __future__ import annotations
import datetime
import operator
import warnings
from unittest import mock
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import xarray as xr
from xarray import DataArray, Dataset, Variable
from xarray.core.groupby import _consolidate_slices
from xarray.core.types import InterpOptions
from xarray.tests import (
@pytest.mark.filterwarnings('ignore:Converting non-nanosecond')
@pytest.mark.filterwarnings('ignore:invalid value encountered in divide:RuntimeWarning')
def test_groupby_drops_nans() -> None:
    ds = xr.Dataset({'variable': (('lat', 'lon', 'time'), np.arange(60.0).reshape((4, 3, 5))), 'id': (('lat', 'lon'), np.arange(12.0).reshape((4, 3)))}, coords={'lat': np.arange(4), 'lon': np.arange(3), 'time': np.arange(5)})
    ds['id'].values[0, 0] = np.nan
    ds['id'].values[3, 0] = np.nan
    ds['id'].values[-1, -1] = np.nan
    grouped = ds.groupby(ds.id)
    expected1 = ds.copy()
    expected1.variable.values[0, 0, :] = np.nan
    expected1.variable.values[-1, -1, :] = np.nan
    expected1.variable.values[3, 0, :] = np.nan
    actual1 = grouped.map(lambda x: x).transpose(*ds.variable.dims)
    assert_identical(actual1, expected1)
    actual2 = grouped.mean()
    stacked = ds.stack({'xy': ['lat', 'lon']})
    expected2 = stacked.variable.where(stacked.id.notnull()).rename({'xy': 'id'}).to_dataset().reset_index('id', drop=True).assign(id=stacked.id.values).dropna('id').transpose(*actual2.variable.dims)
    assert_identical(actual2, expected2)
    actual3 = grouped.mean('time')
    expected3 = ds.mean('time').where(ds.id.notnull())
    assert_identical(actual3, expected3)
    array = xr.DataArray([1, 2, 3], [('x', [1, 2, 3])])
    array['x1'] = ('x', [1, 1, np.nan])
    expected4 = xr.DataArray(3, [('x1', [1])])
    actual4 = array.groupby('x1').sum()
    assert_equal(expected4, actual4)
    array['t'] = ('x', [np.datetime64('2001-01-01'), np.datetime64('2001-01-01'), np.datetime64('NaT')])
    expected5 = xr.DataArray(3, [('t', [np.datetime64('2001-01-01')])])
    actual5 = array.groupby('t').sum()
    assert_equal(expected5, actual5)
    array = xr.DataArray([0, 1, 2, 4, 3, 4], [('x', [np.nan, 1, 1, np.nan, 2, np.nan])])
    expected6 = xr.DataArray([3, 3], [('x', [1, 2])])
    actual6 = array.groupby('x').sum()
    assert_equal(expected6, actual6)