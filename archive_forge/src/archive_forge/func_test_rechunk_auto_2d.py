from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_rechunk_auto_2d():
    x = da.ones((20, 20), chunks=(2, 2))
    y = x.rechunk({0: -1, 1: 'auto'}, block_size_limit=20 * x.dtype.itemsize)
    assert y.chunks == ((20,), (1,) * 20)
    x = da.ones((20, 20), chunks=(2, 2))
    y = x.rechunk((-1, 'auto'), block_size_limit=80 * x.dtype.itemsize)
    assert y.chunks == ((20,), (4,) * 5)
    x = da.ones((20, 20), chunks=(2, 2))
    y = x.rechunk({0: 'auto'}, block_size_limit=20 * x.dtype.itemsize)
    assert y.chunks[1] == x.chunks[1]
    assert y.chunks[0] == (10, 10)
    x = da.ones((20, 20), chunks=((2,) * 10, (2, 2, 2, 2, 2, 5, 5)))
    y = x.rechunk({0: 'auto'}, block_size_limit=20 * x.dtype.itemsize)
    assert y.chunks[1] == x.chunks[1]
    assert y.chunks[0] == (4, 4, 4, 4, 4)