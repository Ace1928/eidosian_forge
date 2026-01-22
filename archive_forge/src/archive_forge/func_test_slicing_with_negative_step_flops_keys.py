from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
def test_slicing_with_negative_step_flops_keys():
    x = da.arange(10, chunks=5)
    y = x[:1:-1]
    assert (x.name, 1) in y.dask[y.name, 0]
    assert (x.name, 0) in y.dask[y.name, 1]
    assert_eq(y, np.arange(10)[:1:-1])
    assert y.chunks == ((5, 3),)
    assert y.dask[y.name, 0] == (getitem, (x.name, 1), (slice(-1, -6, -1),))
    assert y.dask[y.name, 1] == (getitem, (x.name, 0), (slice(-1, -4, -1),))