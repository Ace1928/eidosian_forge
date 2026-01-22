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
def test_multiple_list_slicing():
    x = np.random.default_rng().random((6, 7, 8))
    a = da.from_array(x, chunks=(3, 3, 3))
    assert_eq(x[:, [0, 1, 2]][[0, 1]], a[:, [0, 1, 2]][[0, 1]])