from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_gufunc_mixed_inputs():

    def foo(x, y):
        return x + y
    a = np.ones((2, 1), dtype=int)
    b = da.ones((1, 8), chunks=(2, 3), dtype=int)
    x = apply_gufunc(foo, '(),()->()', a, b, output_dtypes=int)
    assert_eq(x, 2 * np.ones((2, 8), dtype=int))