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
def test_scheduler_keyword():

    def schedule(dsk, keys, **kwargs):
        return [[123]]
    named_schedulers['foo'] = schedule
    x = delayed(inc)(1)
    try:
        assert x.compute() == 2
        assert x.compute(scheduler='foo') == 123
        with dask.config.set(scheduler='foo'):
            assert x.compute() == 123
        assert x.compute() == 2
        with dask.config.set(scheduler='foo'):
            assert x.compute(scheduler='threads') == 2
    finally:
        del named_schedulers['foo']