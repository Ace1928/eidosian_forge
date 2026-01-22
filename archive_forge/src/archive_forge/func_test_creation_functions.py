from __future__ import annotations
import random
import sys
from copy import deepcopy
from itertools import product
import numpy as np
import pytest
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, ComplexWarning
from dask.array.utils import assert_eq
from dask.base import tokenize
from dask.utils import typename
def test_creation_functions():
    x = np.array([-2, -1, 0, 1, 2] * 20).reshape((10, 10))
    y = np.array([-2, 0, 1, 1, 0] * 2)
    dx = da.from_array(x, chunks=5)
    dy = da.from_array(y, chunks=4)
    sol = np.ma.masked_greater(x, y)
    for a, b in product([dx, x], [dy, y]):
        assert_eq(da.ma.masked_greater(a, b), sol)
    assert_eq(da.ma.masked_greater(dx, 0), np.ma.masked_greater(x, 0))
    assert_eq(da.ma.masked_greater_equal(dx, 0), np.ma.masked_greater_equal(x, 0))
    assert_eq(da.ma.masked_less(dx, 0), np.ma.masked_less(x, 0))
    assert_eq(da.ma.masked_less_equal(dx, 0), np.ma.masked_less_equal(x, 0))
    assert_eq(da.ma.masked_equal(dx, 0), np.ma.masked_equal(x, 0))
    assert_eq(da.ma.masked_not_equal(dx, 0), np.ma.masked_not_equal(x, 0))
    assert_eq(da.ma.masked_where(False, dx), np.ma.masked_where(False, x))
    assert_eq(da.ma.masked_where(dx > 2, dx), np.ma.masked_where(x > 2, x))
    with pytest.raises(IndexError):
        da.ma.masked_where((dx > 2)[:, 0], dx)
    assert_eq(da.ma.masked_inside(dx, -1, 1), np.ma.masked_inside(x, -1, 1))
    assert_eq(da.ma.masked_outside(dx, -1, 1), np.ma.masked_outside(x, -1, 1))
    assert_eq(da.ma.masked_values(dx, -1), np.ma.masked_values(x, -1))
    with pytest.raises(ValueError):
        da.ma.masked_equal(dx, dy)
    with pytest.raises(ValueError):
        da.ma.masked_values(dx, dy)
    y = x.astype('f8')
    y[0, 0] = y[7, 5] = np.nan
    dy = da.from_array(y, chunks=5)
    assert_eq(da.ma.masked_invalid(dy), np.ma.masked_invalid(y))
    my = np.ma.masked_greater(y, 0)
    dmy = da.ma.masked_greater(dy, 0)
    assert_eq(da.ma.fix_invalid(dmy, fill_value=0), np.ma.fix_invalid(my, fill_value=0))