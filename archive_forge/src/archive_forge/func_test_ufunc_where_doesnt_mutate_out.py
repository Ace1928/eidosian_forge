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
def test_ufunc_where_doesnt_mutate_out():
    """Dask array's are immutable, ensure that the backing numpy array for
    `out` isn't actually mutated"""
    left = da.from_array(np.arange(4, dtype='i8'), chunks=2)
    right = da.from_array(np.arange(4, 8, dtype='i8'), chunks=2)
    where = da.from_array(np.array([1, 0, 0, 1], dtype='bool'), chunks=2)
    out_np = np.zeros(4, dtype='i8')
    out = da.from_array(out_np, chunks=2)
    result = da.add(left, right, where=where, out=out)
    assert out is result
    assert_eq(out, np.array([4, 0, 0, 10], dtype='i8'))
    assert np.equal(out_np, 0).all()