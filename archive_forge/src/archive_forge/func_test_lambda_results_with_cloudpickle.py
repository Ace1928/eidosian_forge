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
def test_lambda_results_with_cloudpickle():
    dsk = {'x': (lambda_result,)}
    f = get(dsk, 'x')
    assert f(2) == 3