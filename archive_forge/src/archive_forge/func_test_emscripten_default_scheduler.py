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
@pytest.mark.parametrize('params', ("'dask.dataframe', '_Frame', 'sync', True", "'dask.dataframe', '_Frame', 'threads', False", "'dask.array', 'Array', 'sync', True", "'dask.array', 'Array', 'threads', False", "'dask.bag', 'Bag', 'sync', True", "'dask.bag', 'Bag', 'processes', False"))
def test_emscripten_default_scheduler(params):
    pytest.importorskip('dask.array')
    dd = pytest.importorskip('dask.dataframe')
    if dd._dask_expr_enabled() and 'dask.dataframe' in params:
        pytest.skip('objects not available')
    proc = subprocess.run([sys.executable, '-c', inspect.getsource(check_default_scheduler) + f'check_default_scheduler({params})\n'])
    proc.check_returncode()