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
@pytest.mark.skipif('not da')
def test_compute_array_bag():
    x = da.arange(5, chunks=2)
    b = db.from_sequence([1, 2, 3])
    pytest.raises(ValueError, lambda: compute(x, b))
    xx, bb = compute(x, b, scheduler='single-threaded')
    assert np.allclose(xx, np.arange(5))
    assert bb == [1, 2, 3]