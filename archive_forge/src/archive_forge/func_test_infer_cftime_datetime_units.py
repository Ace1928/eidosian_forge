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
@pytest.mark.parametrize('calendar', _NON_STANDARD_CALENDARS + ['gregorian', 'proleptic_gregorian'])
@pytest.mark.parametrize(('date_args', 'expected'), _CFTIME_DATETIME_UNITS_TESTS)
def test_infer_cftime_datetime_units(calendar, date_args, expected) -> None:
    date_type = _all_cftime_date_types()[calendar]
    dates = [date_type(*args) for args in date_args]
    assert expected == coding.times.infer_datetime_units(dates)