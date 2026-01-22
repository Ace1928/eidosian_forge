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
def test_histogram_return_type():
    v = da.random.default_rng().random(100, chunks=10)
    bins = np.arange(0, 1.01, 0.01)
    bins = np.arange(0, 11, 1, dtype='i4')
    assert_eq(da.histogram(v * 10, bins=bins)[0], np.histogram(v * 10, bins=bins)[0])