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
def test_atop_legacy():
    x = da.ones(10, chunks=(5,))
    with pytest.warns(UserWarning, match='The da.atop function has moved to da.blockwise'):
        y = da.atop(inc, 'i', x, 'i', dtype=x.dtype)
    z = da.blockwise(inc, 'i', x, 'i', dtype=x.dtype)
    assert_eq(y, z)
    assert y.name == z.name