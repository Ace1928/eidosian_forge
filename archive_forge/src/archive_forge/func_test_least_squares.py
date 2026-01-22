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
@pytest.mark.parametrize('use_dask', [True, False])
@pytest.mark.parametrize('skipna', [True, False])
def test_least_squares(use_dask, skipna):
    if use_dask and (not has_dask or not has_scipy):
        pytest.skip('requires dask and scipy')
    lhs = np.array([[1, 2], [1, 2], [3, 2]])
    rhs = DataArray(np.array([3, 5, 7]), dims=('y',))
    if use_dask:
        rhs = rhs.chunk({'y': 1})
    coeffs, residuals = least_squares(lhs, rhs.data, skipna=skipna)
    np.testing.assert_allclose(coeffs, [1.5, 1.25])
    np.testing.assert_allclose(residuals, [2.0])