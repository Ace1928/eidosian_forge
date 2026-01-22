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
def test_slice_list_then_None():
    x = da.zeros(shape=(5, 5), chunks=(3, 3))
    y = x[[2, 1]][None]
    assert_eq(y, np.zeros((1, 2, 5)))