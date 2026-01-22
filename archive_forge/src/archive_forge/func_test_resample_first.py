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
def test_resample_first(self) -> None:
    times = pd.date_range('2000-01-01', freq='6h', periods=10)
    array = DataArray(np.arange(10), [('time', times)])
    actual = array.resample(time='6h').first()
    assert_identical(array, actual)
    actual = array.resample(time='1D').first()
    expected = DataArray([0, 4, 8], [('time', times[::4])])
    assert_identical(expected, actual)
    actual = array.resample(time='24h').first()
    expected = DataArray(array.to_series().resample('24h').first())
    assert_identical(expected, actual)
    array = array.astype(float)
    array[:2] = np.nan
    actual = array.resample(time='1D').first()
    expected = DataArray([2, 4, 8], [('time', times[::4])])
    assert_identical(expected, actual)
    actual = array.resample(time='1D').first(skipna=False)
    expected = DataArray([np.nan, 4, 8], [('time', times[::4])])
    assert_identical(expected, actual)
    array = Dataset({'time': times})['time']
    actual = array.resample(time='1D').last()
    expected_times = pd.to_datetime(['2000-01-01T18', '2000-01-02T18', '2000-01-03T06'])
    expected = DataArray(expected_times, [('time', times[::4])], name='time')
    assert_identical(expected, actual)