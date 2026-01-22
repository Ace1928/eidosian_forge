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
def test_groupby_bins_timeseries() -> None:
    ds = xr.Dataset()
    ds['time'] = xr.DataArray(pd.date_range('2010-08-01', '2010-08-15', freq='15min'), dims='time')
    ds['val'] = xr.DataArray(np.ones(ds['time'].shape), dims='time')
    time_bins = pd.date_range(start='2010-08-01', end='2010-08-15', freq='24h')
    actual = ds.groupby_bins('time', time_bins).sum()
    expected = xr.DataArray(96 * np.ones((14,)), dims=['time_bins'], coords={'time_bins': pd.cut(time_bins, time_bins).categories}).to_dataset(name='val')
    assert_identical(actual, expected)