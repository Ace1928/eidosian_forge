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
@pytest.mark.parametrize('calendar', _STANDARD_CALENDARS)
def test_decode_standard_calendar_multidim_time_inside_timestamp_range(calendar) -> None:
    import cftime
    units = 'days since 0001-01-01'
    times1 = pd.date_range('2001-04-01', end='2001-04-05', freq='D')
    times2 = pd.date_range('2001-05-01', end='2001-05-05', freq='D')
    time1 = cftime.date2num(times1.to_pydatetime(), units, calendar=calendar)
    time2 = cftime.date2num(times2.to_pydatetime(), units, calendar=calendar)
    mdim_time = np.empty((len(time1), 2))
    mdim_time[:, 0] = time1
    mdim_time[:, 1] = time2
    expected1 = times1.values
    expected2 = times2.values
    actual = coding.times.decode_cf_datetime(mdim_time, units, calendar=calendar)
    assert actual.dtype == np.dtype('M8[ns]')
    abs_diff1 = abs(actual[:, 0] - expected1)
    abs_diff2 = abs(actual[:, 1] - expected2)
    assert (abs_diff1 <= np.timedelta64(1, 's')).all()
    assert (abs_diff2 <= np.timedelta64(1, 's')).all()