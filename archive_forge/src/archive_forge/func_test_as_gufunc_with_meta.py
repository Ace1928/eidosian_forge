from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_as_gufunc_with_meta():
    stack = da.ones((1, 50, 60), chunks=(1, -1, -1))
    expected = (stack, stack.max())
    meta = (np.array((), dtype=np.float64), np.array((), dtype=np.float64))

    @da.as_gufunc(signature='(i,j) ->(i,j), ()', meta=meta)
    def array_and_max(arr):
        return (arr, np.atleast_1d(arr.max()))
    result = array_and_max(stack)
    assert_eq(expected[0], result[0])
    assert_eq(np.array([expected[1].compute()]), result[1].compute())