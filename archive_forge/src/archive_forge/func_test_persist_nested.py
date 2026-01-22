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
def test_persist_nested(c):
    a = delayed(1) + 5
    b = a + 1
    c = a + 2
    result = persist({'a': a, 'b': [1, 2, b]}, (c, 2), 4, [5])
    assert isinstance(result[0]['a'], Delayed)
    assert isinstance(result[0]['b'][2], Delayed)
    assert isinstance(result[1][0], Delayed)
    sol = ({'a': 6, 'b': [1, 2, 7]}, (8, 2), 4, [5])
    assert compute(*result) == sol
    res = persist([a, b], c, 4, [5], traverse=False)
    assert res[0][0] is a
    assert res[0][1] is b
    assert res[1].compute() == 8
    assert res[2:] == (4, [5])