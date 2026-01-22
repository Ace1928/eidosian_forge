from __future__ import annotations
import pickle
import warnings
from functools import partial
from operator import add
import pytest
import dask.array as da
from dask.array.ufunc import da_frompyfunc
from dask.array.utils import assert_eq
from dask.base import tokenize
def test_out_numpy():
    x = da.arange(10, chunks=(5,))
    empty = np.empty(10, dtype=x.dtype)
    with pytest.raises((TypeError, NotImplementedError)) as info:
        np.add(x, 1, out=empty)
    assert 'ndarray' in str(info.value)
    assert 'Array' in str(info.value)