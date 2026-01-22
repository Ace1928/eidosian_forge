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
@pytest.mark.parametrize('freq', ['us', 'ms', 's', 'min', 'h', 'D'])
def test_encode_decode_roundtrip_cftime(freq) -> None:
    initial_time = cftime_range('0001', periods=1)
    times = initial_time.append(cftime_range('0001', periods=2, freq=freq) + timedelta(days=291000 * 365))
    variable = Variable(['time'], times)
    encoded = conventions.encode_cf_variable(variable)
    decoded = conventions.decode_cf_variable('time', encoded, use_cftime=True)
    assert_equal(variable, decoded)