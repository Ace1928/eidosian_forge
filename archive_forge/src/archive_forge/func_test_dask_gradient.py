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
@pytest.mark.skipif(not has_dask, reason='This is for dask.')
@pytest.mark.parametrize('axis', [0, -1, 1])
@pytest.mark.parametrize('edge_order', [1, 2])
def test_dask_gradient(axis, edge_order):
    import dask.array as da
    array = np.array(np.random.randn(100, 5, 40))
    x = np.exp(np.linspace(0, 1, array.shape[axis]))
    darray = da.from_array(array, chunks=[(6, 30, 30, 20, 14), 5, 8])
    expected = gradient(array, x, axis=axis, edge_order=edge_order)
    actual = gradient(darray, x, axis=axis, edge_order=edge_order)
    assert isinstance(actual, da.Array)
    assert_array_equal(actual, expected)