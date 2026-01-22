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
@requires_dask
@requires_bottleneck
def test_push_dask():
    import bottleneck
    import dask.array
    array = np.array([np.nan, 1, 2, 3, np.nan, np.nan, np.nan, np.nan, 4, 5, np.nan, 6])
    for n in [None, 1, 2, 3, 4, 5, 11]:
        expected = bottleneck.push(array, axis=0, n=n)
        for c in range(1, 11):
            with raise_if_dask_computes():
                actual = push(dask.array.from_array(array, chunks=c), axis=0, n=n)
            np.testing.assert_equal(actual, expected)
        with raise_if_dask_computes():
            actual = push(dask.array.from_array(array, chunks=(1, 2, 3, 2, 2, 1, 1)), axis=0, n=n)
        np.testing.assert_equal(actual, expected)