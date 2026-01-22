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
@pytest.mark.parametrize('calendar', ['gregorian', 'Gregorian', 'GREGORIAN'])
def test_decode_encode_roundtrip_with_non_lowercase_letters(calendar) -> None:
    times = [0, 1]
    units = 'days since 2000-01-01'
    attrs = {'calendar': calendar, 'units': units}
    variable = Variable(['time'], times, attrs)
    decoded = conventions.decode_cf_variable('time', variable)
    encoded = conventions.encode_cf_variable(decoded)
    assert np.issubdtype(decoded.dtype, np.datetime64)
    assert_identical(variable, encoded)