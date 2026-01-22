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
def test_argwhere():
    for shape, chunks in [(0, ()), ((0, 0), (0, 0)), ((15, 16), (4, 5))]:
        x = np.random.default_rng().integers(10, size=shape)
        d = da.from_array(x, chunks=chunks)
        x_nz = np.argwhere(x)
        d_nz = da.argwhere(d)
        assert_eq(d_nz, x_nz)