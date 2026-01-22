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
@pytest.mark.filterwarnings("ignore:Times can't be serialized faithfully")
@pytest.mark.parametrize('encoding_units', FREQUENCIES_TO_ENCODING_UNITS.values())
@pytest.mark.parametrize('freq', FREQUENCIES_TO_ENCODING_UNITS.keys())
@pytest.mark.parametrize('date_range', [pd.date_range, cftime_range])
def test_encode_cf_datetime_defaults_to_correct_dtype(encoding_units, freq, date_range) -> None:
    if not has_cftime and date_range == cftime_range:
        pytest.skip('Test requires cftime')
    if (freq == 'ns' or encoding_units == 'nanoseconds') and date_range == cftime_range:
        pytest.skip('Nanosecond frequency is not valid for cftime dates.')
    times = date_range('2000', periods=3, freq=freq)
    units = f'{encoding_units} since 2000-01-01'
    encoded, _units, _ = coding.times.encode_cf_datetime(times, units)
    numpy_timeunit = coding.times._netcdf_to_numpy_timeunit(encoding_units)
    encoding_units_as_timedelta = np.timedelta64(1, numpy_timeunit)
    if pd.to_timedelta(1, freq) >= encoding_units_as_timedelta:
        assert encoded.dtype == np.int64
    else:
        assert encoded.dtype == np.float64