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
@pytest.mark.parametrize(('calendar', 'num_time'), [('360_day', 720058.0), ('all_leap', 732059.0), ('366_day', 732059.0)])
def test_decode_non_standard_calendar_single_element(calendar, num_time) -> None:
    import cftime
    units = 'days since 0001-01-01'
    actual = coding.times.decode_cf_datetime(num_time, units, calendar=calendar)
    expected = np.asarray(cftime.num2date(num_time, units, calendar, only_use_cftime_datetimes=True))
    assert actual.dtype == np.dtype('O')
    assert expected == actual