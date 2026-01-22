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
def test_persist_array_bag():
    x = da.arange(5, chunks=2) + 1
    b = db.from_sequence([1, 2, 3]).map(inc)
    with pytest.raises(ValueError):
        persist(x, b)
    xx, bb = persist(x, b, scheduler='single-threaded')
    assert isinstance(xx, da.Array)
    assert isinstance(bb, db.Bag)
    assert xx.name == x.name
    assert bb.name == b.name
    assert len(xx.dask) == xx.npartitions < len(x.dask)
    assert len(bb.dask) == bb.npartitions < len(b.dask)
    assert np.allclose(x, xx)
    assert list(b) == list(bb)