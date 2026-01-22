from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
@pytest.mark.parametrize('op', [_shuffle_op, _groupby_op])
def test_dataframe_cull_key_dependencies(op):
    dd = pytest.importorskip('dask.dataframe')
    if dd._dask_expr_enabled():
        pytest.skip('not supported')
    datasets = pytest.importorskip('dask.datasets')
    result = op(datasets.timeseries(end='2000-01-15')).count()
    graph = result.dask
    culled_graph = graph.cull(result.__dask_keys__())
    assert graph.get_all_dependencies() == culled_graph.get_all_dependencies()