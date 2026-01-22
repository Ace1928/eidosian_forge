from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
def test_array_slice_deps():
    dac = pytest.importorskip('dask.array.core')
    d = 2
    chunk = (2, 3)
    shape = tuple((d * n for n in chunk))
    chunks = dac.normalize_chunks(chunk, shape)
    array_deps = ArraySliceDep(chunks)

    def check(i, j):
        slices = array_deps[i, j]
        assert slices == (slice(chunk[0] * i, chunk[0] * (i + 1), None), slice(chunk[1] * j, chunk[1] * (j + 1), None))
    for i in range(d):
        for j in range(d):
            check(i, j)