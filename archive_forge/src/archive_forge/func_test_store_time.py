from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
def test_store_time():
    p = ProgressBar()
    with p:
        get_threaded({'x': 1}, 'x')
    assert isinstance(p.last_duration, float)