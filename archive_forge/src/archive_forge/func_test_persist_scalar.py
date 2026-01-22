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
@pytest.mark.skipif('not dd')
def test_persist_scalar():
    import dask.dataframe as dd
    ds = pd.Series([1, 2, 3, 4])
    dds1 = dd.from_pandas(ds, npartitions=2).min()
    assert len(dds1.__dask_graph__()) == 5 if not dd._dask_expr_enabled() else 6
    dds2 = dds1.persist()
    if not dd._dask_expr_enabled():
        assert isinstance(dds2, dd.core.Scalar)
    assert len(dds2.__dask_graph__()) == 1 if not dd._dask_expr_enabled() else 2
    dd.utils.assert_eq(dds2, dds1)