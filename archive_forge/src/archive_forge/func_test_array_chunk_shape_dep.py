from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
def test_array_chunk_shape_dep():
    dac = pytest.importorskip('dask.array.core')
    d = 2
    chunk = (2, 3)
    shape = tuple((d * n for n in chunk))
    chunks = dac.normalize_chunks(chunk, shape)
    array_deps = ArrayChunkShapeDep(chunks)

    def check(i, j):
        chunk_shape = array_deps[i, j]
        assert chunk_shape == chunk
    for i in range(d):
        for j in range(d):
            check(i, j)