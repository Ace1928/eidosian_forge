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
def test_encode_cf_datetime_casting_value_error(use_cftime, use_dask) -> None:
    times = date_range(start='2000', freq='12h', periods=3, use_cftime=use_cftime)
    encoding = dict(units='days since 2000-01-01', dtype=np.dtype('int64'))
    variable = Variable(['time'], times, encoding=encoding)
    if use_dask:
        variable = variable.chunk({'time': 1})
    if not use_cftime and (not use_dask):
        with pytest.warns(UserWarning, match="Times can't be serialized"):
            encoded = conventions.encode_cf_variable(variable)
        assert encoded.attrs['units'] == 'hours since 2000-01-01'
        decoded = conventions.decode_cf_variable('name', encoded)
        assert_equal(variable, decoded)
    else:
        with pytest.raises(ValueError, match='Not possible'):
            encoded = conventions.encode_cf_variable(variable)
            encoded.compute()