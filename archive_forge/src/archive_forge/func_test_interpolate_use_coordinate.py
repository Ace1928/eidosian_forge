from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def test_interpolate_use_coordinate():
    xc = xr.Variable('x', [100, 200, 300, 400, 500, 600])
    da = xr.DataArray(np.array([1, 2, np.nan, np.nan, np.nan, 6], dtype=np.float64), dims='x', coords={'xc': xc})
    actual = da.interpolate_na(dim='x', use_coordinate=False)
    expected = da.interpolate_na(dim='x')
    assert_equal(actual, expected)
    actual = da.interpolate_na(dim='x', use_coordinate='xc')
    expected = da.interpolate_na(dim='x')
    assert_equal(actual, expected)
    actual = da.interpolate_na(dim='x', use_coordinate='x')
    expected = da.interpolate_na(dim='x')
    assert_equal(actual, expected)