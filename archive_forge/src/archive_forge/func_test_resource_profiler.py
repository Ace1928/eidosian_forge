from __future__ import annotations
import contextlib
import os
import warnings
from operator import add, mul
from timeit import default_timer
import pytest
from dask.diagnostics import CacheProfiler, Profiler, ResourceProfiler
from dask.diagnostics.profile_visualize import BOKEH_VERSION
from dask.threaded import get
from dask.utils import apply, tmpfile
from dask.utils_test import slowadd
@pytest.mark.skipif('not psutil')
def test_resource_profiler():
    with ResourceProfiler(dt=0.01) as rprof:
        in_context_time = default_timer()
        get(dsk2, 'c')
    results = rprof.results
    assert len(results) > 0
    assert all((isinstance(i, tuple) and len(i) == 3 for i in results))
    assert rprof.start_time < in_context_time < rprof.end_time
    assert not rprof._is_running()
    rprof.clear()
    assert rprof.results == []
    rprof.close()
    assert not rprof._is_running()
    with rprof:
        get(dsk2, 'c')
    assert len(rprof.results) > 0