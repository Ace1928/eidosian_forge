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
def test_histogram_extra_args_and_shapes():
    bins = np.arange(0, 1.01, 0.01)
    v = da.random.default_rng().random(100, chunks=10)
    data = [(v, bins, da.ones(100, chunks=v.chunks) * 5), (da.random.default_rng().random((50, 50), chunks=10), bins, da.ones((50, 50), chunks=10) * 5)]
    for v, bins, w in data:
        assert_eq(da.histogram(v, bins=bins, density=True)[0], np.histogram(v, bins=bins, density=True)[0])
        assert_eq(da.histogram(v, bins=bins, weights=w)[0], np.histogram(v, bins=bins, weights=w)[0])
        assert_eq(da.histogram(v, bins=bins, weights=w, density=True)[0], da.histogram(v, bins=bins, weights=w, density=True)[0])