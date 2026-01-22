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
@pytest.mark.parametrize('io', ['ones', 'zeros', 'full'])
@pytest.mark.parametrize('fuse', [True, False, None])
def test_blockwise_array_creation(c, io, fuse):
    np = pytest.importorskip('numpy')
    da = pytest.importorskip('dask.array')
    chunks = (5, 2)
    shape = (10, 4)
    if io == 'ones':
        darr = da.ones(shape, chunks=chunks)
        narr = np.ones(shape)
    elif io == 'zeros':
        darr = da.zeros(shape, chunks=chunks)
        narr = np.zeros(shape)
    elif io == 'full':
        darr = da.full(shape, 10, chunks=chunks)
        narr = np.full(shape, 10)
    darr += 2
    narr += 2
    with dask.config.set({'optimization.fuse.active': fuse}):
        darr.compute()
        dsk = dask.array.optimize(darr.dask, darr.__dask_keys__())
        assert isinstance(dsk, dict) == (fuse is not False)
        da.assert_eq(darr, narr, scheduler=c)