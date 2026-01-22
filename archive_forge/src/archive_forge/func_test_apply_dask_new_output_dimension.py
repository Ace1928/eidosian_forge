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
def test_apply_dask_new_output_dimension() -> None:
    import dask.array as da
    array = da.ones((2, 2), chunks=(1, 1))
    data_array = xr.DataArray(array, dims=('x', 'y'))

    def stack_negative(obj):

        def func(x):
            return np.stack([x, -x], axis=-1)
        return apply_ufunc(func, obj, output_core_dims=[['sign']], dask='parallelized', output_dtypes=[obj.dtype], dask_gufunc_kwargs=dict(output_sizes={'sign': 2}))
    expected = stack_negative(data_array.compute())
    actual = stack_negative(data_array)
    assert actual.dims == ('x', 'y', 'sign')
    assert actual.shape == (2, 2, 2)
    assert isinstance(actual.data, da.Array)
    assert_identical(expected, actual)