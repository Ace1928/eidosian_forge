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
def construct_dataarray(dim_num, dtype, contains_nan, dask):
    rng = np.random.RandomState(0)
    shapes = [16, 8, 4][:dim_num]
    dims = ('x', 'y', 'z')[:dim_num]
    if np.issubdtype(dtype, np.floating):
        array = rng.randn(*shapes).astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        array = rng.randint(0, 10, size=shapes).astype(dtype)
    elif np.issubdtype(dtype, np.bool_):
        array = rng.randint(0, 1, size=shapes).astype(dtype)
    elif dtype == str:
        array = rng.choice(['a', 'b', 'c', 'd'], size=shapes)
    else:
        raise ValueError
    if contains_nan:
        inds = rng.choice(range(array.size), int(array.size * 0.2))
        dtype, fill_value = dtypes.maybe_promote(array.dtype)
        array = array.astype(dtype)
        array.flat[inds] = fill_value
    da = DataArray(array, dims=dims, coords={'x': np.arange(16)}, name='da')
    if dask and has_dask:
        chunks = {d: 4 for d in dims}
        da = da.chunk(chunks)
    return da