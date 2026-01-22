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
def test_empty_slice():
    x = da.ones((5, 5), chunks=(2, 2), dtype='i4')
    y = x[:0]
    assert_eq(y, np.ones((5, 5), dtype='i4')[:0])