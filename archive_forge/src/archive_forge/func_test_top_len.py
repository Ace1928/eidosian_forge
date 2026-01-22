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
def test_top_len():
    x = da.ones(10, chunks=(5,))
    y = x[:, None] * x[None, :]
    d = y.dask.layers[y.name]
    assert len(d) == 4