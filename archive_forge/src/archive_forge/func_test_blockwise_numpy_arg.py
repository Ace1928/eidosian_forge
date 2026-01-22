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
def test_blockwise_numpy_arg():
    x = da.arange(10, chunks=(5,))
    y = np.arange(1000)
    x = x.map_blocks(lambda x, y: x, 1.0)
    x = x.map_blocks(lambda x, y: x, 'abc')
    x = x.map_blocks(lambda x, y: x, y)
    x = x.map_blocks(lambda x, y: x, 'abc')
    x = x.map_blocks(lambda x, y: x, 1.0)
    x = x.map_blocks(lambda x, y, z: x, 'abc', np.array(['a', 'b'], dtype=object))
    assert_eq(x, np.arange(10))