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
def test_optimize_graph_false():
    from dask.callbacks import Callback
    d = {'x': 1, 'y': (inc, 'x'), 'z': (add, 10, 'y')}
    keys = []
    with Callback(pretask=lambda key, *args: keys.append(key)):
        get(d, 'z', optimize_graph=False)
    assert len(keys) == 2