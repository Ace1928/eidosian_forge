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
def test_unpicklable_args_generate_errors():
    a = NotUnpickleable()
    dsk = {'x': (bool, a)}
    with pytest.raises(ValueError):
        get(dsk, 'x')
    dsk = {'x': (bool, 'a'), 'a': a}
    with pytest.raises(ValueError):
        get(dsk, 'x')