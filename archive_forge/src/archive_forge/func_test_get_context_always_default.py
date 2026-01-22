from __future__ import annotations
import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from operator import add
import pytest
import dask
from dask import compute, delayed
from dask.multiprocessing import _dumps, _loads, get, get_context, remote_exception
from dask.system import CPU_COUNT
from dask.utils_test import inc
@pytest.mark.skipif(sys.platform != 'win32', reason='POSIX supports different contexts')
def test_get_context_always_default():
    """On Python 2/Windows, get_context() always returns same context."""
    assert get_context() is multiprocessing
    with pytest.warns(UserWarning):
        with dask.config.set({'multiprocessing.context': 'forkserver'}):
            assert get_context() is multiprocessing