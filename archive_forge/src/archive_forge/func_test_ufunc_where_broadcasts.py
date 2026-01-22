from __future__ import annotations
import pickle
import warnings
from functools import partial
from operator import add
import pytest
import dask.array as da
from dask.array.ufunc import da_frompyfunc
from dask.array.utils import assert_eq
from dask.base import tokenize
@pytest.mark.parametrize('left_is_da', [False, True])
@pytest.mark.parametrize('right_is_da', [False, True])
@pytest.mark.parametrize('where_is_da', [False, True])
def test_ufunc_where_broadcasts(left_is_da, right_is_da, where_is_da):
    left = np.arange(4)
    right = np.arange(4, 8)
    where = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]]).astype('bool')
    out = np.zeros(where.shape, dtype=left.dtype)
    d_out = da.zeros(where.shape, dtype=left.dtype)
    d_where = da.from_array(where, chunks=2) if where_is_da else where
    d_left = da.from_array(left, chunks=2) if left_is_da else left
    d_right = da.from_array(right, chunks=2) if right_is_da else right
    expected = np.add(left, right, where=where, out=out)
    result = da.add(d_left, d_right, where=d_where, out=d_out)
    assert result is d_out
    assert_eq(expected, result)