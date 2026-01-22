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
def test_slicing_consistent_names():
    x = np.arange(100).reshape((10, 10))
    a = da.from_array(x, chunks=(5, 5))
    assert same_keys(a[0], a[0])
    assert same_keys(a[:, [1, 2, 3]], a[:, [1, 2, 3]])
    assert same_keys(a[:, 5:2:-1], a[:, 5:2:-1])
    assert same_keys(a[0, ...], a[0, ...])
    assert same_keys(a[...], a[...])
    assert same_keys(a[[1, 3, 5]], a[[1, 3, 5]])
    assert same_keys(a[-11:11], a[:])
    assert same_keys(a[-11:-9], a[:1])
    assert same_keys(a[-1], a[9])
    assert same_keys(a[0::-1], a[0:-11:-1])