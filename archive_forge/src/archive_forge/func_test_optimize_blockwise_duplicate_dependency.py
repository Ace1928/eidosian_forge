from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.chunk import getitem as da_getitem
from dask.array.core import getter as da_getter
from dask.array.core import getter_nofancy as da_getter_nofancy
from dask.array.optimization import (
from dask.array.utils import assert_eq
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse
from dask.utils import SerializableLock
@pytest.mark.parametrize('optimize_graph', [True, False])
def test_optimize_blockwise_duplicate_dependency(optimize_graph):
    xx = da.from_array(np.array([[1, 1], [2, 2]]), chunks=1)
    xx = xx * 2
    z = da.matmul(xx, xx)
    result = z.compute(optimize_graph=optimize_graph)
    assert assert_eq(result, [[12, 12], [24, 24]])