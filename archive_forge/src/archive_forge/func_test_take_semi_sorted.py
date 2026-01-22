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
def test_take_semi_sorted():
    x = da.ones(10, chunks=(5,))
    index = np.arange(15) % 10
    y = x[index]
    assert y.chunks == ((5, 5, 5),)