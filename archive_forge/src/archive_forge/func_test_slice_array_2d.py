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
def test_slice_array_2d():
    expected = {('y', 0, 0): (getitem, ('x', 0, 0), (slice(13, 20, 2), slice(10, 20, 1))), ('y', 0, 1): (getitem, ('x', 0, 1), (slice(13, 20, 2), slice(None, None, None))), ('y', 0, 2): (getitem, ('x', 0, 2), (slice(13, 20, 2), slice(None, None, None)))}
    result, chunks = slice_array('y', 'x', [[20], [20, 20, 5]], [slice(13, None, 2), slice(10, None, 1)], itemsize=8)
    assert expected == result
    expected = {('y', 0): (getitem, ('x', 0, 0), (5, slice(10, 20, 1))), ('y', 1): (getitem, ('x', 0, 1), (5, slice(None, None, None))), ('y', 2): (getitem, ('x', 0, 2), (5, slice(None, None, None)))}
    result, chunks = slice_array('y', 'x', ([20], [20, 20, 5]), [5, slice(10, None, 1)], 8)
    assert expected == result