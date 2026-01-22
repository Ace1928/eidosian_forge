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
def test_dot_persist_equivalence():
    x = da.random.default_rng().random((4, 4), chunks=(2, 2))
    x[x < 0.65] = 0
    y = x.persist()
    z = x.compute()
    r1 = da.dot(x, x).compute()
    r2 = da.dot(y, y).compute()
    rr = np.dot(z, z)
    assert np.allclose(rr, r1)
    assert np.allclose(rr, r2)