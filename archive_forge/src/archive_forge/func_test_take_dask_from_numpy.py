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
def test_take_dask_from_numpy():
    x = np.arange(5).astype('f8')
    y = da.from_array(np.array([1, 2, 3, 3, 2, 1]), chunks=3)
    z = da.take(x * 2, y)
    assert z.chunks == y.chunks
    assert_eq(z, np.array([2.0, 4.0, 6.0, 6.0, 4.0, 2.0]))