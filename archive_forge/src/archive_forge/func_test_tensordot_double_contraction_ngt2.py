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
def test_tensordot_double_contraction_ngt2():
    x = np.arange(60.0).reshape(3, 4, 5)
    y = np.arange(60.0).reshape(4, 5, 3)
    u = da.from_array(x)
    v = da.from_array(y)
    assert_eq(da.tensordot(u, v, axes=2), np.tensordot(x, y, axes=2))
    x = np.arange(60.0).reshape(3, 4, 5)
    y = np.arange(60.0).reshape(4, 5, 3)
    u = da.from_array(x, chunks=3)
    v = da.from_array(y)
    assert_eq(da.tensordot(u, v, axes=2), np.tensordot(x, y, axes=2))