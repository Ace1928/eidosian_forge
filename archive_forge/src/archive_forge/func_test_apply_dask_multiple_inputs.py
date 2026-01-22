from __future__ import annotations
import functools
import operator
import pickle
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import xarray as xr
from xarray.core.alignment import broadcast
from xarray.core.computation import (
from xarray.tests import (
@requires_dask
@pytest.mark.filterwarnings('ignore:Mean of empty slice')
def test_apply_dask_multiple_inputs() -> None:
    import dask.array as da

    def covariance(x, y):
        return ((x - x.mean(axis=-1, keepdims=True)) * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)
    rs = np.random.RandomState(42)
    array1 = da.from_array(rs.randn(4, 4), chunks=(2, 4))
    array2 = da.from_array(rs.randn(4, 4), chunks=(2, 4))
    data_array_1 = xr.DataArray(array1, dims=('x', 'z'))
    data_array_2 = xr.DataArray(array2, dims=('y', 'z'))
    expected = apply_ufunc(covariance, data_array_1.compute(), data_array_2.compute(), input_core_dims=[['z'], ['z']])
    allowed = apply_ufunc(covariance, data_array_1, data_array_2, input_core_dims=[['z'], ['z']], dask='allowed')
    assert isinstance(allowed.data, da.Array)
    xr.testing.assert_allclose(expected, allowed.compute())
    parallelized = apply_ufunc(covariance, data_array_1, data_array_2, input_core_dims=[['z'], ['z']], dask='parallelized', output_dtypes=[float])
    assert isinstance(parallelized.data, da.Array)
    xr.testing.assert_allclose(expected, parallelized.compute())