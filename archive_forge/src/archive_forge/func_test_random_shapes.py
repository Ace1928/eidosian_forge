from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask import config
from dask.array.utils import assert_eq
@pytest.mark.parametrize('shape', [(2, 3), (2, 3, 4), (2, 3, 4, 2)])
def test_random_shapes(shape):
    rs = da.random.RandomState(RandomState=cupy.random.RandomState)
    x = rs.poisson(size=shape, chunks=3)
    assert type(x._meta) == cupy.ndarray
    assert_eq(x, x)
    assert x._meta.shape == (0,) * len(shape)
    assert x.shape == shape
    rng = da.random.default_rng(cupy.random.default_rng())
    x = rng.poisson(1.0, size=shape, chunks=3)
    assert type(x._meta) == cupy.ndarray
    assert_eq(x, x)
    assert x._meta.shape == (0,) * len(shape)
    assert x.shape == shape