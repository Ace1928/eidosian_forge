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
def test_optimize_None():
    da = pytest.importorskip('dask.array')
    x = da.ones(10, chunks=(5,))
    y = x[:9][1:8][::2] + 1

    def my_get(dsk, keys):
        assert dsk == dict(y.dask)
        return dask.get(dsk, keys)
    with dask.config.set(array_optimize=None, scheduler=my_get):
        y.compute()