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
def test_errors_propagate():
    dsk = {'x': (bad,)}
    with pytest.raises(ValueError) as e:
        get(dsk, 'x')
    assert '12345' in str(e.value)