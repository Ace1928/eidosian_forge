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
@requires_cftime
def test_datetime_to_numeric_potential_overflow():
    import cftime
    times = pd.date_range('2000', periods=5, freq='7D').values.astype('datetime64[us]')
    cftimes = cftime_range('2000', periods=5, freq='7D', calendar='proleptic_gregorian').values
    offset = np.datetime64('0001-01-01')
    cfoffset = cftime.DatetimeProlepticGregorian(1, 1, 1)
    result = duck_array_ops.datetime_to_numeric(times, offset=offset, datetime_unit='D', dtype=int)
    cfresult = duck_array_ops.datetime_to_numeric(cftimes, offset=cfoffset, datetime_unit='D', dtype=int)
    expected = 730119 + np.arange(0, 35, 7)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(cfresult, expected)