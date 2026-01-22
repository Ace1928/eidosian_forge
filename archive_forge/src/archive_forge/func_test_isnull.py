from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
@pytest.mark.parametrize(['array', 'expected'], [(np.array([np.datetime64('2000-01-01'), np.datetime64('NaT')]), np.array([False, True])), (np.array([np.timedelta64(1, 'h'), np.timedelta64('NaT')]), np.array([False, True])), (np.array([0.0, np.nan]), np.array([False, True])), (np.array([1j, np.nan]), np.array([False, True])), (np.array(['foo', np.nan], dtype=object), np.array([False, True])), (np.array([1, 2], dtype=int), np.array([False, False])), (np.array([True, False], dtype=bool), np.array([False, False]))])
def test_isnull(array, expected):
    actual = duck_array_ops.isnull(array)
    np.testing.assert_equal(expected, actual)