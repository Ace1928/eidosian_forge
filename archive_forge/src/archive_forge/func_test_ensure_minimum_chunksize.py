from __future__ import annotations
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import dask.array as da
from dask.array.lib.stride_tricks import sliding_window_view
from dask.array.overlap import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('chunks,expected', [[(10,), (10,)], [(10, 10), (10, 10)], [(10, 10, 1), (10, 11)], [(20, 20, 20, 1), (20, 20, 11, 10)], [(20, 20, 10, 1), (20, 20, 11)], [(2, 20, 2, 20), (14, 10, 20)], [(1, 1, 1, 1, 7), (11,)], [(20, 20, 2, 20, 20, 2), (20, 12, 10, 20, 12, 10)]])
def test_ensure_minimum_chunksize(chunks, expected):
    actual = ensure_minimum_chunksize(10, chunks)
    assert actual == expected