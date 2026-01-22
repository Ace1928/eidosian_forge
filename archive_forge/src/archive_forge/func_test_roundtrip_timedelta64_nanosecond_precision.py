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
@pytest.mark.parametrize('dtype, fill_value', [(np.int64, 20), (np.int64, np.iinfo(np.int64).min), (np.float64, 1e+30)])
def test_roundtrip_timedelta64_nanosecond_precision(dtype: np.typing.DTypeLike, fill_value: int | float) -> None:
    one_day = np.timedelta64(1, 'ns')
    nat = np.timedelta64('nat', 'ns')
    timedelta_values = (np.arange(5) * one_day).astype('timedelta64[ns]')
    timedelta_values[2] = nat
    timedelta_values[4] = nat
    encoding = dict(dtype=dtype, _FillValue=fill_value)
    var = Variable(['time'], timedelta_values, encoding=encoding)
    encoded_var = conventions.encode_cf_variable(var)
    decoded_var = conventions.decode_cf_variable('foo', encoded_var)
    assert_identical(var, decoded_var)