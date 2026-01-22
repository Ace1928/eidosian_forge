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
def test_multi_insert():
    z = np.random.default_rng().integers(10, size=(1, 2))
    c = da.from_array(z, chunks=(1, 2))
    assert_eq(np.insert(np.insert(z, [0, 1], -1, axis=0), [1], -1, axis=1), da.insert(da.insert(c, [0, 1], -1, axis=0), [1], -1, axis=1))