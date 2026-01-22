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
@requires_cftime
@requires_scipy
def test_cftime_single_string() -> None:
    from cftime import DatetimeProlepticGregorian
    times = xr.cftime_range('2000', periods=24, freq='D', calendar='proleptic_gregorian')
    da = xr.DataArray(np.arange(24), coords=[times], dims='time')
    times_new = '2000-01-01T12:00'
    actual = da.interp(time=times_new)
    times_new_array = _parse_array_of_cftime_strings(np.array(times_new), DatetimeProlepticGregorian)
    expected = xr.DataArray(0.5, coords={'time': times_new_array})
    assert_allclose(actual, expected)