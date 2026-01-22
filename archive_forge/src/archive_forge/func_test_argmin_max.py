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
@pytest.mark.parametrize('dim_num', [1, 2])
@pytest.mark.parametrize('dtype', [float, int, np.float32, np.bool_, str])
@pytest.mark.parametrize('contains_nan', [True, False])
@pytest.mark.parametrize('dask', [False, True])
@pytest.mark.parametrize('func', ['min', 'max'])
@pytest.mark.parametrize('skipna', [False, True])
@pytest.mark.parametrize('aggdim', ['x', 'y'])
def test_argmin_max(dim_num, dtype, contains_nan, dask, func, skipna, aggdim):
    if aggdim == 'y' and dim_num < 2:
        pytest.skip('dim not in this test')
    if dask and (not has_dask):
        pytest.skip('requires dask')
    if contains_nan:
        if not skipna:
            pytest.skip("numpy's argmin (not nanargmin) does not handle object-dtype")
        if skipna and np.dtype(dtype).kind in 'iufc':
            pytest.skip("numpy's nanargmin raises ValueError for all nan axis")
    da = construct_dataarray(dim_num, dtype, contains_nan=contains_nan, dask=dask)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice')
        actual = da.isel(**{aggdim: getattr(da, 'arg' + func)(dim=aggdim, skipna=skipna).compute()})
        expected = getattr(da, func)(dim=aggdim, skipna=skipna)
        assert_allclose(actual.drop_vars(list(actual.coords)), expected.drop_vars(list(expected.coords)))