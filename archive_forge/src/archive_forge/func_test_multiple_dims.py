from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('func', ['sum', 'prod'])
def test_multiple_dims(dtype, dask, skipna, func):
    if dask and (not has_dask):
        pytest.skip('requires dask')
    da = construct_dataarray(3, dtype, contains_nan=True, dask=dask)
    actual = getattr(da, func)(('x', 'y'), skipna=skipna)
    expected = getattr(getattr(da, func)('x', skipna=skipna), func)('y', skipna=skipna)
    assert_allclose(actual, expected)