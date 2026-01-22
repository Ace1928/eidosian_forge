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
def test_apply_dask_parallelized_one_arg() -> None:
    import dask.array as da
    array = da.ones((2, 2), chunks=(1, 1))
    data_array = xr.DataArray(array, dims=('x', 'y'))

    def parallel_identity(x):
        return apply_ufunc(identity, x, dask='parallelized', output_dtypes=[x.dtype])
    actual = parallel_identity(data_array)
    assert isinstance(actual.data, da.Array)
    assert actual.data.chunks == array.chunks
    assert_identical(data_array, actual)
    computed = data_array.compute()
    actual = parallel_identity(computed)
    assert_identical(computed, actual)