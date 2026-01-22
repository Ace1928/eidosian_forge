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
def test_blockwise_new_axes_2():
    x = da.ones((2, 2), chunks=(1, 1))

    def func(x):
        return np.stack([x, -x], axis=-1)
    y = da.blockwise(func, ('x', 'y', 'sign'), x, ('x', 'y'), dtype=x.dtype, concatenate=True, new_axes={'sign': 2})
    assert_eq(y, y)