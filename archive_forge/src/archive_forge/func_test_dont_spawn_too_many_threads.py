from __future__ import annotations
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
from time import sleep, time
import pytest
import dask
from dask.system import CPU_COUNT
from dask.threaded import get
from dask.utils_test import add, inc
def test_dont_spawn_too_many_threads():
    before = threading.active_count()
    dsk = {('x', i): (lambda i=i: i,) for i in range(10)}
    dsk['x'] = (sum, list(dsk))
    for _ in range(20):
        get(dsk, 'x', num_workers=4)
    after = threading.active_count()
    assert after <= before + 8