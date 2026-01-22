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
def test_histogramdd_weighted():
    rng = da.random.default_rng()
    n1, n2 = (600, 3)
    x = rng.uniform(0, 1, size=(n1, n2), chunks=((200, 200, 200), (3,)))
    w = rng.uniform(0.5, 0.8, size=(n1,), chunks=200)
    bins = (3, 5, 4)
    ranges = ((0, 1),) * len(bins)
    a1, b1 = da.histogramdd(x, bins=bins, range=ranges, weights=w)
    a2, b2 = np.histogramdd(x, bins=bins, range=ranges, weights=w)
    a3, b3 = np.histogramdd(x.compute(), bins=bins, range=ranges, weights=w.compute())
    assert_eq(a1, a2)
    assert_eq(a1, a3)
    bins = 4
    a1, b1 = da.histogramdd(x, bins=bins, range=ranges, weights=w)
    a2, b2 = np.histogramdd(x, bins=bins, range=ranges, weights=w)
    a3, b3 = np.histogramdd(x.compute(), bins=bins, range=ranges, weights=w.compute())
    assert_eq(a1, a2)
    assert_eq(a1, a3)