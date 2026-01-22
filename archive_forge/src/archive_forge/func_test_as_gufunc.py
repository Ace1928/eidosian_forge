from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_as_gufunc():
    x = da.random.default_rng().normal(size=(10, 5), chunks=(2, 5))

    @as_gufunc('(i)->()', axis=-1, keepdims=False, output_dtypes=float, vectorize=True)
    def foo(x):
        return np.mean(x, axis=-1)
    y = foo(x)
    valy = y.compute()
    assert isinstance(y, Array)
    assert valy.shape == (10,)