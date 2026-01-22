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
@requires_cftime
def test_time_units_with_timezone_roundtrip(calendar) -> None:
    expected_units = 'days since 2000-01-01T00:00:00-05:00'
    expected_num_dates = np.array([1, 2, 3])
    dates = decode_cf_datetime(expected_num_dates, expected_units, calendar)
    result_hours = DataArray(dates).dt.hour
    expected_hours = DataArray([5, 5, 5])
    assert_equal(result_hours, expected_hours)
    result_num_dates, result_units, result_calendar = encode_cf_datetime(dates, expected_units, calendar)
    if calendar in _STANDARD_CALENDARS:
        np.testing.assert_array_equal(result_num_dates, expected_num_dates)
    else:
        np.testing.assert_allclose(result_num_dates, expected_num_dates)
    assert result_units == expected_units
    assert result_calendar == calendar