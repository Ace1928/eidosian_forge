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
def test_count_nonzero_obj():
    x = np.random.default_rng().integers(10, size=(15, 16)).astype(object)
    d = da.from_array(x, chunks=(4, 5))
    x_c = np.count_nonzero(x)
    d_c = da.count_nonzero(d)
    if d_c.shape == tuple():
        assert x_c == d_c.compute()
    else:
        assert_eq(x_c, d_c)