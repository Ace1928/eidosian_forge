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
@pytest.mark.parametrize('use_cftime', [True, False])
def test_resample_doctest(self, use_cftime: bool) -> None:
    if use_cftime and (not has_cftime):
        pytest.skip()
    da = xr.DataArray(np.array([1, 2, 3, 1, 2, np.nan]), dims='time', coords=dict(time=('time', xr.date_range('2001-01-01', freq='ME', periods=6, use_cftime=use_cftime)), labels=('time', np.array(['a', 'b', 'c', 'c', 'b', 'a']))))
    actual = da.resample(time='3ME').count()
    expected = DataArray([1, 3, 1], dims='time', coords={'time': xr.date_range('2001-01-01', freq='3ME', periods=3, use_cftime=use_cftime)})
    assert_identical(actual, expected)