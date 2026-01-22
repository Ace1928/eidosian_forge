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
def test_blockwise_diamond_fusion():
    x = da.ones(10, chunks=(5,))
    y = x + 1 + 2 + 3
    a = y * 2
    b = y * 3
    c = a + b
    d = c + 1 + 2 + 3
    dsk = da.optimization.optimize_blockwise(d.dask)
    assert isinstance(dsk, HighLevelGraph)
    assert len([layer for layer in dsk.layers.values() if isinstance(layer, Blockwise)]) == 1