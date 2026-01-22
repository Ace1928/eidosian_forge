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
@pytest.mark.skipif('not bokeh')
def test_saves_file_path_deprecated():
    with tmpfile('html') as fn:
        with prof:
            get(dsk, 'e')
        with pytest.warns(FutureWarning) as record:
            prof.visualize(show=False, file_path=fn)
        assert 1 <= len(record) <= 2
        assert 'file_path keyword argument is deprecated' in str(record[-1].message)
        if len(record) == 2:
            assert '`np.bool8` is a deprecated alias for `np.bool_`' in str(record[0].message)