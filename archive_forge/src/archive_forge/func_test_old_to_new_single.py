from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_old_to_new_single():
    old = ((np.nan, np.nan), (8,))
    new = ((np.nan, np.nan), (4, 4))
    result = old_to_new(old, new)
    expected = [[[(0, slice(0, None, None))], [(1, slice(0, None, None))]], [[(0, slice(0, 4, None))], [(0, slice(4, 8, None))]]]
    assert result == expected