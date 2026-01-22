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
def test_is_dask_collection_dask_expr_does_not_materialize():
    dx = pytest.importorskip('dask_expr')

    class DoNotMaterialize(dx._core.Expr):

        @property
        def _meta(self):
            return 0

        def __dask_keys__(self):
            assert False, 'must not reach'

        def __dask_graph__(self):
            assert False, 'must not reach'

        def optimize(self, fuse=False):
            assert False, 'must not reach'
    coll = dx._collection.new_collection(DoNotMaterialize())
    with pytest.raises(AssertionError, match='must not reach'):
        coll.__dask_keys__()
    with pytest.raises(AssertionError, match='must not reach'):
        coll.__dask_graph__()
    assert is_dask_collection(coll)