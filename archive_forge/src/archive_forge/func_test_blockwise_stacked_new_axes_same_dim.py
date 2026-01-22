from __future__ import annotations
import collections
from operator import add
import numpy as np
import pytest
import dask
import dask.array as da
from dask.array.utils import assert_eq
from dask.blockwise import (
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import dec, hlg_layer_topological, inc
@pytest.mark.parametrize('concatenate', [True, False])
def test_blockwise_stacked_new_axes_same_dim(concatenate):

    def f(x):
        return x[..., None] * np.ones((1, 7))
    x = da.ones(5, chunks=2)
    y = da.zeros(5, chunks=2)
    a = da.blockwise(f, 'aq', x, 'a', new_axes={'q': 7}, concatenate=concatenate, dtype=x.dtype)
    b = da.blockwise(f, 'aq', y, 'a', new_axes={'q': 7}, concatenate=concatenate, dtype=x.dtype)
    c = a + b
    assert c.chunks == ((2, 2, 1), (7,))
    assert_eq(c, np.ones((5, 7)))