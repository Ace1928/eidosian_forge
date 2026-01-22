from __future__ import annotations
from operator import add, mul
import pytest
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from dask.diagnostics.progress import format_time
from dask.local import get_sync
from dask.threaded import get as get_threaded
def test_minimum_time(capsys):
    with ProgressBar(10.0):
        out = get_threaded(dsk, 'e')
    out, err = capsys.readouterr()
    assert out == '' and err == ''