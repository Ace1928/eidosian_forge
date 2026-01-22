from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
def test_pickle_vectorized_routines():
    """Test that graphs that internally use np.vectorize can be pickled"""
    a = da.from_array(['foo', 'bar', ''])
    b = da.count_nonzero(a)
    assert_eq(b, 2, check_dtype=False)
    b2 = pickle.loads(pickle.dumps(b))
    assert_eq(b2, 2, check_dtype=False)
    c = da.argwhere(a)
    assert_eq(c, [[0], [1]], check_dtype=False)
    c2 = pickle.loads(pickle.dumps(c))
    assert_eq(c2, [[0], [1]], check_dtype=False)