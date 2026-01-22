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
@pytest.mark.skipif('not psutil')
def test_plot_multiple():
    from dask.diagnostics.profile_visualize import visualize
    with ResourceProfiler(dt=0.01) as rprof:
        with prof:
            get(dsk2, 'c')
    p = visualize([prof, rprof], label_size=50, title='Not the default', show=False, save=False)
    if BOKEH_VERSION().major < 3:
        figures = [r[0] for r in p.children[1].children]
    else:
        figures = [r[0] for r in p.children]
    assert len(figures) == 2
    assert figures[0].title.text == 'Not the default'
    assert figures[0].xaxis[0].axis_label is None
    assert figures[1].title is None
    assert figures[1].xaxis[0].axis_label == 'Time (s)'
    prof.clear()
    rprof.clear()
    visualize([prof, rprof], show=False, save=False)