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
def test_slice_array_null_dimension():
    array = da.from_array(np.zeros((3, 0)))
    expected = np.zeros((3, 0))[[0]]
    assert_eq(array[[0]], expected)