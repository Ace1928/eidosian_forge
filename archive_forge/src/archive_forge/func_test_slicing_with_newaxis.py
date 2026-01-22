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
def test_slicing_with_newaxis():
    result, chunks = slice_array('y', 'x', ([5, 5], [5, 5]), (slice(0, 3), None, slice(None, None, None)), itemsize=8)
    expected = {('y', 0, 0, 0): (getitem, ('x', 0, 0), (slice(0, 3, 1), None, slice(None, None, None))), ('y', 0, 0, 1): (getitem, ('x', 0, 1), (slice(0, 3, 1), None, slice(None, None, None)))}
    assert expected == result
    assert chunks == ((3,), (1,), (5, 5))