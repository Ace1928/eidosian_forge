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
def test_array_ufunc():
    x = np.arange(24).reshape((4, 6))
    d = da.from_array(x, chunks=(2, 3))
    for func in [np.sin, np.sum, np.negative, partial(np.prod, axis=0)]:
        assert isinstance(func(d), da.Array)
        assert_eq(func(d), func(x))