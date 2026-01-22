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
@pytest.mark.parametrize('shape,chunks,bs,expected', [(100, 1, 10, (10,) * 10), (100, 50, 10, (10,) * 10), (100, 100, 10, (10,) * 10), (20, 7, 10, (7, 7, 6)), (20, (1, 1, 1, 1, 6, 2, 1, 7), 5, (5, 5, 5, 5))])
def test_rechunk_auto_1d(shape, chunks, bs, expected):
    x = da.ones(shape, chunks=(chunks,))
    y = x.rechunk({0: 'auto'}, block_size_limit=bs * x.dtype.itemsize)
    assert y.chunks == (expected,)