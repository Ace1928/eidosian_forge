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
def test_balance_chunks_unchanged():
    arr_len = 220
    x = da.from_array(np.arange(arr_len))
    balanced = x.rechunk(chunks=100, balance=True)
    unbalanced = x.rechunk(chunks=100, balance=False)
    assert unbalanced.chunks[0] == (100, 100, 20)
    assert balanced.chunks[0] == (110, 110)