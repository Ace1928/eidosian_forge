from __future__ import annotations
import pytest
import operator
import numpy as np
import dask.array as da
from dask.array.chunk import coarsen, getitem, keepdims_wrapper
def test_coarsen_unaligned_shape():
    """https://github.com/dask/dask/issues/10274"""
    x = da.random.random(100)
    res = da.coarsen(np.mean, x, {0: 3}, trim_excess=True)
    assert res.chunks == ((33,),)