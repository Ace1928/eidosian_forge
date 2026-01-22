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
def test_balance_n_chunks_size():
    arr_len = 100
    n_chunks = 8
    x = da.from_array(np.arange(arr_len))
    balanced = x.rechunk(chunks=arr_len // n_chunks, balance=True)
    unbalanced = x.rechunk(chunks=arr_len // n_chunks, balance=False)
    assert balanced.chunks[0] == (13,) * 7 + (9,)
    assert unbalanced.chunks[0] == (12,) * 8 + (4,)