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
def test_apply_dask_parallelized_two_args() -> None:
    import dask.array as da
    array = da.ones((2, 2), chunks=(1, 1), dtype=np.int64)
    data_array = xr.DataArray(array, dims=('x', 'y'))
    data_array.name = None

    def parallel_add(x, y):
        return apply_ufunc(operator.add, x, y, dask='parallelized', output_dtypes=[np.int64])

    def check(x, y):
        actual = parallel_add(x, y)
        assert isinstance(actual.data, da.Array)
        assert actual.data.chunks == array.chunks
        assert_identical(data_array, actual)
    check(data_array, 0)
    check(0, data_array)
    check(data_array, xr.DataArray(0))
    check(data_array, 0 * data_array)
    check(data_array, 0 * data_array[0])
    check(data_array[:, 0], 0 * data_array[0])
    check(data_array, 0 * data_array.compute())