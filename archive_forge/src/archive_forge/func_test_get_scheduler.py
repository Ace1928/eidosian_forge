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
def test_get_scheduler():
    assert get_scheduler() is None
    assert get_scheduler(scheduler=dask.local.get_sync) is dask.local.get_sync
    assert get_scheduler(scheduler='threads') is dask.threaded.get
    assert get_scheduler(scheduler='sync') is dask.local.get_sync
    assert callable(get_scheduler(scheduler=dask.local.synchronous_executor))
    assert callable(get_scheduler(scheduler=MyExecutor()))
    with dask.config.set(scheduler='threads'):
        assert get_scheduler() is dask.threaded.get
    assert get_scheduler() is None