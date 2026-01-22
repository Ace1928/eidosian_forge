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
def test_slice_array_3d_with_bool_numpy_array():
    array = da.arange(0, 24).reshape((4, 3, 2))
    mask = np.arange(0, 24).reshape((4, 3, 2)) > 12
    actual = array[mask].compute()
    expected = np.arange(13, 24)
    assert_eq(actual, expected)