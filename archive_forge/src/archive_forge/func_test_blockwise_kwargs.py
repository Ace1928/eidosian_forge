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
def test_blockwise_kwargs():

    def f(a, b=0):
        return a + b
    x = da.ones(5, chunks=(2,))
    y = da.blockwise(f, 'i', x, 'i', b=10, dtype=x.dtype)
    assert_eq(y, np.ones(5) + 10)