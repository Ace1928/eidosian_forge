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
@pytest.mark.parametrize('calendar', _ALL_CALENDARS)
def test_decode_dates_outside_timestamp_range(calendar) -> None:
    from datetime import datetime
    import cftime
    units = 'days since 0001-01-01'
    times = [datetime(1, 4, 1, h) for h in range(1, 5)]
    time = cftime.date2num(times, units, calendar=calendar)
    expected = cftime.num2date(time, units, calendar=calendar, only_use_cftime_datetimes=True)
    expected_date_type = type(expected[0])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Unable to decode time axis')
        actual = coding.times.decode_cf_datetime(time, units, calendar=calendar)
    assert all((isinstance(value, expected_date_type) for value in actual))
    abs_diff = abs(actual - expected)
    assert (abs_diff <= np.timedelta64(1, 's')).all()