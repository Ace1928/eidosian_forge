from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
@pytest.mark.flaky(reruns=10, reruns_delay=5)
@pytest.mark.slow
@pytest.mark.parametrize('scheduler', ['threads', 'processes'])
def test_num_workers_config(scheduler):
    f = delayed(pure=False)(time.sleep)
    num_workers = 3
    a = [f(1.0) for i in range(num_workers)]
    with dask.config.set(num_workers=num_workers, chunksize=1), Profiler() as prof:
        compute(*a, scheduler=scheduler)
    workers = {i.worker_id for i in prof.results}
    assert len(workers) == num_workers