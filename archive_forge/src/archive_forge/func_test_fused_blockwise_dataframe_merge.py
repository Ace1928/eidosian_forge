from __future__ import annotations
import pytest
import asyncio
import os
import sys
from functools import partial
from operator import add
from distributed import Client, SchedulerPlugin, WorkerPlugin
from distributed.utils_test import cleanup  # noqa F401
from distributed.utils_test import client as c  # noqa F401
from distributed.utils_test import (  # noqa F401
import dask
import dask.bag as db
from dask import compute, delayed, persist
from dask.base import compute_as_if_collection, get_scheduler
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.distributed import futures_of, wait
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.utils import get_named_args, get_scheduler_lock, tmpdir, tmpfile
from dask.utils_test import inc
@pytest.mark.parametrize('fuse', [True, False])
def test_fused_blockwise_dataframe_merge(c, fuse):
    pd = pytest.importorskip('pandas')
    dd = pytest.importorskip('dask.dataframe')
    size = 35
    df1 = pd.DataFrame({'x': range(size), 'y': range(size)})
    df2 = pd.DataFrame({'x': range(size), 'z': range(size)})
    ddf1 = dd.from_pandas(df1, npartitions=size) + 10
    ddf2 = dd.from_pandas(df2, npartitions=5) + 10
    df1 += 10
    df2 += 10
    with dask.config.set({'optimization.fuse.active': fuse}):
        ddfm = ddf1.merge(ddf2, on=['x'], how='left', shuffle_method='tasks')
        ddfm.head()
        dfm = ddfm.compute().sort_values('x')
    dd.utils.assert_eq(dfm, df1.merge(df2, on=['x'], how='left').sort_values('x'), check_index=False)