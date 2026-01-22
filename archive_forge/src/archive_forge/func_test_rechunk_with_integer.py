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
def test_rechunk_with_integer():
    x = da.from_array(np.arange(5), chunks=4)
    y = x.rechunk(3)
    assert y.chunks == ((3, 2),)
    assert (x.compute() == y.compute()).all()