from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
def test_fractional_slice():
    assert fractional_slice(('x', 4.9), {0: 2}) == (getitem, ('x', 5), (slice(0, 2),))
    assert fractional_slice(('x', 3, 5.1), {0: 2, 1: 3}) == (getitem, ('x', 3, 5), (slice(None, None, None), slice(-3, None)))
    assert fractional_slice(('x', 2.9, 5.1), {0: 2, 1: 3}) == (getitem, ('x', 3, 5), (slice(0, 2), slice(-3, None)))
    fs = fractional_slice(('x', 4.9), {0: 2})
    assert isinstance(fs[1][1], int)