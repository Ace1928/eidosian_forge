import os
import sys
import time
import mmap
import weakref
import warnings
import threading
from traceback import format_exception
from math import sqrt
from time import sleep
from pickle import PicklingError
from contextlib import nullcontext
from multiprocessing import TimeoutError
import pytest
import joblib
from joblib import parallel
from joblib import dump, load
from joblib._multiprocessing_helpers import mp
from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.test.common import IS_PYPY, force_gc_pypy
from joblib.testing import (parametrize, raises, check_subprocess_call,
from queue import Queue
from joblib._parallel_backends import SequentialBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib._parallel_backends import ParallelBackendBase
from joblib._parallel_backends import LokyBackend
from joblib.parallel import Parallel, delayed
from joblib.parallel import parallel_config
from joblib.parallel import parallel_backend
from joblib.parallel import register_parallel_backend
from joblib.parallel import effective_n_jobs, cpu_count
from joblib.parallel import mp, BACKENDS, DEFAULT_BACKEND
from joblib import Parallel, delayed
import sys
from joblib import Parallel, delayed
import sys
import faulthandler
from joblib import Parallel, delayed
from functools import partial
import sys
from joblib import Parallel, delayed, hash
import multiprocessing as mp
@with_multiprocessing
@parametrize('context', [parallel_config, parallel_backend])
def test_backend_hinting_and_constraints(context):
    for n_jobs in [1, 2, -1]:
        assert type(Parallel(n_jobs=n_jobs)._backend) == DefaultBackend
        p = Parallel(n_jobs=n_jobs, prefer='threads')
        assert type(p._backend) is ThreadingBackend
        p = Parallel(n_jobs=n_jobs, prefer='processes')
        assert type(p._backend) is DefaultBackend
        p = Parallel(n_jobs=n_jobs, require='sharedmem')
        assert type(p._backend) is ThreadingBackend
    p = Parallel(n_jobs=2, backend='loky', prefer='threads')
    assert type(p._backend) is LokyBackend
    with context('loky', n_jobs=2):
        p = Parallel(prefer='threads')
        assert type(p._backend) is LokyBackend
        assert p.n_jobs == 2
    with context('loky', n_jobs=2):
        p = Parallel(n_jobs=3, prefer='threads')
        assert type(p._backend) is LokyBackend
        assert p.n_jobs == 3
    with context('loky', n_jobs=2):
        p = Parallel(require='sharedmem')
        assert type(p._backend) is ThreadingBackend
        assert p.n_jobs == 1
    with context('loky', n_jobs=2):
        p = Parallel(n_jobs=3, require='sharedmem')
        assert type(p._backend) is ThreadingBackend
        assert p.n_jobs == 3