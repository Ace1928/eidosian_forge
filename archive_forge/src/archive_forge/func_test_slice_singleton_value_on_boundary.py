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
def test_slice_singleton_value_on_boundary():
    assert _slice_1d(15, [5, 5, 5], 10) == {2: 0}
    assert _slice_1d(30, (5, 5, 5, 5, 5, 5), 10) == {2: 0}