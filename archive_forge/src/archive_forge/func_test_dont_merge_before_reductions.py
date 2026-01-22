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
def test_dont_merge_before_reductions():
    x = da.ones(10, chunks=(5,))
    y = da.blockwise(inc, 'i', x, 'i', dtype=x.dtype)
    z = da.blockwise(sum, '', y, 'i', dtype=y.dtype)
    w = da.blockwise(sum, '', z, '', dtype=y.dtype)
    dsk = optimize_blockwise(w.dask)
    assert len([d for d in dsk.layers.values() if isinstance(d, Blockwise)]) == 2
    z.compute()