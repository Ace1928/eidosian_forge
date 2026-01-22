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
def test_slice_stop_0():
    a = da.ones(10, chunks=(10,))[:0].compute()
    b = np.ones(10)[:0]
    assert_eq(a, b)