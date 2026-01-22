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
def test_non_hlg():
    a = da.from_array(np.ones(1, np.float64), chunks=(1,))
    a.dask = dict(a.dask)
    b = da.from_array(np.zeros(1, np.float64), chunks=(1,))
    x = a + b
    assert_eq(x, a)