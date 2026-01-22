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
def test_moveaxis_rollaxis_keyword():
    x = np.random.default_rng().random((10, 12, 7))
    d = da.from_array(x, chunks=(4, 5, 2))
    assert_eq(np.moveaxis(x, destination=1, source=0), da.moveaxis(d, destination=1, source=0))
    assert_eq(np.rollaxis(x, 2), da.rollaxis(d, 2))
    assert isinstance(da.rollaxis(d, 1), da.Array)
    assert_eq(np.rollaxis(x, start=1, axis=2), da.rollaxis(d, start=1, axis=2))