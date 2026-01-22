from __future__ import annotations
import os
import pytest
import sys
from operator import getitem
from distributed import Client, SchedulerPlugin
from distributed.utils_test import cluster, loop  # noqa F401
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArrayChunkShapeDep, ArraySliceDep, fractional_slice
@pytest.mark.xfail(reason='#8480')
@pytest.mark.parametrize('op,lib', [(_dataframe_shuffle, 'pandas.'), (_dataframe_tree_reduction, 'pandas.'), (_dataframe_broadcast_join, 'pandas.'), (_pq_pyarrow, 'pandas.'), (_pq_fastparquet, 'pandas.'), (_read_csv, 'pandas.'), (_array_creation, 'numpy.'), (_array_map_overlap, 'numpy.')])
@pytest.mark.parametrize('optimize_graph', [True, False])
def test_scheduler_highlevel_graph_unpack_import(op, lib, optimize_graph, loop, tmpdir):
    with cluster(scheduler_kwargs={'plugins': [SchedulerImportCheck(lib)]}) as (scheduler, workers):
        with Client(scheduler['address'], loop=loop) as c:
            c.compute(op(tmpdir), optimize_graph=optimize_graph)
            end_modules = c.run_on_scheduler(lambda: set(sys.modules))
            start_modules = c.run_on_scheduler(lambda dask_scheduler: dask_scheduler.plugins[SchedulerImportCheck.name].start_modules)
            new_modules = end_modules - start_modules
            assert not any((module.startswith(lib) for module in start_modules))
            assert not any((module.startswith(lib) for module in new_modules))