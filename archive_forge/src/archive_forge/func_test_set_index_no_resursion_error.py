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
def test_set_index_no_resursion_error(c):
    pytest.importorskip('dask.dataframe')
    try:
        ddf = dask.datasets.timeseries(start='2000-01-01', end='2000-07-01', freq='12h').reset_index().astype({'timestamp': str})
        ddf = ddf.set_index('timestamp', sorted=True)
        ddf.compute()
    except RecursionError:
        pytest.fail('dd.set_index triggered a recursion error')