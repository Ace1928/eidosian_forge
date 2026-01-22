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
@parametrize('backend', [None, 'loky'] if mp is not None else [None])
@skipif(parallel_sum is None, reason='Need OpenMP helper compiled')
def test_parallel_thread_limit(backend):
    results = Parallel(n_jobs=2, backend=backend)((delayed(_run_parallel_sum)() for _ in range(2)))
    expected_num_threads = max(cpu_count() // 2, 1)
    for worker_env_vars, omp_num_threads in results:
        assert omp_num_threads == expected_num_threads
        for name, value in worker_env_vars.items():
            if name.endswith('_THREADS'):
                assert value == str(expected_num_threads)
            else:
                assert name == 'ENABLE_IPC'
                assert value == '1'