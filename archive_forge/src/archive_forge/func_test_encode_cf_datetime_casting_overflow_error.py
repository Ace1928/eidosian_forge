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
@pytest.mark.parametrize('use_cftime', [False, pytest.param(True, marks=requires_cftime)])
@pytest.mark.parametrize('use_dask', [False, pytest.param(True, marks=requires_dask)])
@pytest.mark.parametrize('dtype', [np.dtype('int16'), np.dtype('float16')])
def test_encode_cf_datetime_casting_overflow_error(use_cftime, use_dask, dtype) -> None:
    times = date_range(start='2018', freq='5h', periods=3, use_cftime=use_cftime)
    encoding = dict(units='microseconds since 2018-01-01', dtype=dtype)
    variable = Variable(['time'], times, encoding=encoding)
    if use_dask:
        variable = variable.chunk({'time': 1})
    with pytest.raises(OverflowError, match='Not possible'):
        encoded = conventions.encode_cf_variable(variable)
        encoded.compute()