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
@requires_dask
@pytest.mark.parametrize(('range_function', 'start', 'units', 'dtype'), [(pd.date_range, '2000', None, np.dtype('int32')), (pd.date_range, '2000', 'days since 2000-01-01', None), (pd.timedelta_range, '0D', None, np.dtype('int32')), (pd.timedelta_range, '0D', 'days', None)])
def test_encode_via_dask_cannot_infer_error(range_function, start, units, dtype) -> None:
    values = range_function(start=start, freq='D', periods=3)
    encoding = dict(units=units, dtype=dtype)
    variable = Variable(['time'], values, encoding=encoding).chunk({'time': 1})
    with pytest.raises(ValueError, match='When encoding chunked arrays'):
        conventions.encode_cf_variable(variable)