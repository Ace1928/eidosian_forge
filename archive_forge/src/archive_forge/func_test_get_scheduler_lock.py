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
@pytest.mark.parametrize('scheduler, expected_classes', [(None, ('SerializableLock', 'SerializableLock', 'AcquirerProxy')), ('threads', ('SerializableLock', 'SerializableLock', 'SerializableLock')), ('processes', ('AcquirerProxy', 'AcquirerProxy', 'AcquirerProxy'))])
def test_get_scheduler_lock(scheduler, expected_classes):
    da = pytest.importorskip('dask.array', reason='Requires dask.array')
    db = pytest.importorskip('dask.bag', reason='Requires dask.bag')
    dd = pytest.importorskip('dask.dataframe', reason='Requires dask.dataframe')
    darr = da.ones((100,))
    ddf = dd.from_dask_array(darr, columns=['x'])
    dbag = db.range(100, npartitions=2)
    for collection, expected in zip((ddf, darr, dbag), expected_classes):
        res = get_scheduler_lock(collection, scheduler=scheduler)
        assert res.__class__.__name__ == expected