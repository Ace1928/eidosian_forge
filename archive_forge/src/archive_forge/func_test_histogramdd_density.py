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
def test_histogramdd_density():
    n1, n2 = (800, 3)
    x = da.random.default_rng().uniform(0, 1, size=(n1, n2), chunks=(200, 3))
    bins = [[0, 0.5, 1], [0, 0.25, 0.85, 1], [0, 0.5, 0.8, 1]]
    a1, b1 = da.histogramdd(x, bins=bins, density=True)
    a2, b2 = np.histogramdd(x, bins=bins, density=True)
    a3, b3 = da.histogramdd(x, bins=bins, normed=True)
    a4, b4 = np.histogramdd(x.compute(), bins=bins, density=True)
    assert_eq(a1, a2)
    assert_eq(a1, a3)
    assert_eq(a1, a4)
    assert same_keys(da.histogramdd(x, bins=bins, density=True)[0], a1)