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
def test_balance_raises():
    arr_len = 100
    n_chunks = 11
    x = da.from_array(np.arange(arr_len))
    with pytest.warns(UserWarning, match='Try increasing the chunk size'):
        balanced = x.rechunk(chunks=arr_len // n_chunks, balance=True)
    unbalanced = x.rechunk(chunks=arr_len // n_chunks, balance=False)
    assert balanced.chunks == unbalanced.chunks
    n_chunks = 10
    x.rechunk(chunks=arr_len // n_chunks, balance=True)