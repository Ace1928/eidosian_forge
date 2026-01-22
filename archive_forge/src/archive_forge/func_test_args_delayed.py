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
def test_args_delayed():
    x = da.arange(10, chunks=(5,))
    y = dask.delayed(lambda: 100)()
    z = da.blockwise(add, 'i', x, 'i', y, None, dtype=x.dtype)
    assert_eq(z, np.arange(10) + 100)
    z = da.blockwise(lambda x, y: x + y, 'i', x, 'i', y=y, dtype=x.dtype)
    assert_eq(z, np.arange(10) + 100)