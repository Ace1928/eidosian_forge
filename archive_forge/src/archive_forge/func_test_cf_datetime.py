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
@pytest.mark.filterwarnings('ignore:Ambiguous reference date string')
@pytest.mark.filterwarnings("ignore:Times can't be serialized faithfully")
@pytest.mark.parametrize(['num_dates', 'units', 'calendar'], _CF_DATETIME_TESTS)
def test_cf_datetime(num_dates, units, calendar) -> None:
    import cftime
    expected = cftime.num2date(num_dates, units, calendar, only_use_cftime_datetimes=True)
    min_y = np.ravel(np.atleast_1d(expected))[np.nanargmin(num_dates)].year
    max_y = np.ravel(np.atleast_1d(expected))[np.nanargmax(num_dates)].year
    if min_y >= 1678 and max_y < 2262:
        expected = cftime_to_nptime(expected)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Unable to decode time axis')
        actual = coding.times.decode_cf_datetime(num_dates, units, calendar)
    abs_diff = np.asarray(abs(actual - expected)).ravel()
    abs_diff = pd.to_timedelta(abs_diff.tolist()).to_numpy()
    assert (abs_diff <= np.timedelta64(1, 's')).all()
    encoded, _, _ = coding.times.encode_cf_datetime(actual, units, calendar)
    assert_array_equal(num_dates, np.around(encoded, 1))
    if hasattr(num_dates, 'ndim') and num_dates.ndim == 1 and ('1000' not in units):
        encoded, _, _ = coding.times.encode_cf_datetime(pd.Index(actual), units, calendar)
        assert_array_equal(num_dates, np.around(encoded, 1))