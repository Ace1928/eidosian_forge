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
def test_issignedinf():
    with np.errstate(invalid='ignore', divide='ignore'):
        arr = np.random.randint(-1, 2, size=(20, 20)).astype(float) / 0
    darr = da.from_array(arr, 3)
    assert_eq(np.isneginf(arr), da.isneginf(darr))
    assert_eq(np.isposinf(arr), da.isposinf(darr))