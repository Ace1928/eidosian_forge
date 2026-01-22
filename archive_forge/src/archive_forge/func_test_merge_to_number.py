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
def test_merge_to_number():
    chunks = merge_to_number((10,) * 4, 5)
    assert chunks == (10, 10, 10, 10)
    chunks = merge_to_number((10,) * 4, 4)
    assert chunks == (10, 10, 10, 10)
    chunks = merge_to_number((10,) * 4, 3)
    assert chunks == (20, 10, 10)
    chunks = merge_to_number((10,) * 4, 2)
    assert chunks == (20, 20)
    chunks = merge_to_number((10,) * 4, 1)
    assert chunks == (40,)
    chunks = merge_to_number((10,) * 10, 2)
    assert chunks == (50,) * 2
    chunks = merge_to_number((10,) * 10, 3)
    assert chunks == (40, 30, 30)
    chunks = merge_to_number((5, 1, 1, 15, 10), 4)
    assert chunks == (5, 2, 15, 10)
    chunks = merge_to_number((5, 1, 1, 15, 10), 3)
    assert chunks == (7, 15, 10)
    chunks = merge_to_number((5, 1, 1, 15, 10), 2)
    assert chunks == (22, 10)
    chunks = merge_to_number((5, 1, 1, 15, 10), 1)
    assert chunks == (32,)
    chunks = merge_to_number((1, 1, 1, 1, 3, 1, 1), 6)
    assert chunks == (2, 1, 1, 3, 1, 1)
    chunks = merge_to_number((1, 1, 1, 1, 3, 1, 1), 5)
    assert chunks == (2, 2, 3, 1, 1)
    chunks = merge_to_number((1, 1, 1, 1, 3, 1, 1), 4)
    assert chunks == (2, 2, 3, 2)
    chunks = merge_to_number((1, 1, 1, 1, 3, 1, 1), 3)
    assert chunks == (4, 3, 2)
    chunks = merge_to_number((1, 1, 1, 1, 3, 1, 1), 2)
    assert chunks == (4, 5)
    chunks = merge_to_number((1, 1, 1, 1, 3, 1, 1), 1)
    assert chunks == (9,)