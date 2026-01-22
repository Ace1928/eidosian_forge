from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.uint32, np.uint64])
def test_decode_cf_datetime_uint(dtype):
    units = 'seconds since 2018-08-22T03:23:03Z'
    num_dates = dtype(50)
    result = decode_cf_datetime(num_dates, units)
    expected = np.asarray(np.datetime64('2018-08-22T03:23:53', 'ns'))
    np.testing.assert_equal(result, expected)