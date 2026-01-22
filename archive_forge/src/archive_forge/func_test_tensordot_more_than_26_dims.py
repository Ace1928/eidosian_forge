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
def test_tensordot_more_than_26_dims():
    ndim = 27
    x = np.broadcast_to(1, [2] * ndim)
    dx = da.from_array(x, chunks=-1)
    assert_eq(da.tensordot(dx, dx, ndim), np.array(2 ** ndim))