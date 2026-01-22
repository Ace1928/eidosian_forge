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
def test_rechunk_4d():
    """Try rechunking a random 4d matrix"""
    old = ((5, 5),) * 4
    a = np.random.default_rng().uniform(0, 1, 10000).reshape((10,) * 4)
    x = da.from_array(a, chunks=old)
    new = ((10,),) * 4
    x2 = rechunk(x, chunks=new)
    assert x2.chunks == new
    assert np.all(x2.compute() == a)