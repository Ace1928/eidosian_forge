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
@pytest.mark.parametrize('shape', [(24,), (8, 3), (2, 4, 3)])
def test_encode_cf_datetime_overflow(shape) -> None:
    dates = pd.date_range('2100', periods=24).values.reshape(shape)
    units = 'days since 1800-01-01'
    calendar = 'standard'
    num, _, _ = encode_cf_datetime(dates, units, calendar)
    roundtrip = decode_cf_datetime(num, units, calendar)
    np.testing.assert_array_equal(dates, roundtrip)