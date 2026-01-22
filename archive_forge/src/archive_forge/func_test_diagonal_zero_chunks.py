from __future__ import annotations
import pytest
import numpy as np
import pytest
from tlz import concat
import dask
import dask.array as da
from dask.array.core import normalize_chunks
from dask.array.numpy_compat import AxisError
from dask.array.utils import assert_eq, same_keys
def test_diagonal_zero_chunks():
    x = da.ones((8, 8), chunks=(4, 4))
    dd = da.ones((8, 8), chunks=(4, 4))
    d = da.diagonal(dd)
    expected = np.ones((8,))
    assert_eq(d, expected)
    assert_eq(d + d, 2 * expected)
    A = d + x
    assert_eq(A, np.full((8, 8), 2.0))