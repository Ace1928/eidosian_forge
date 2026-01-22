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
@parametrize('context', [parallel_config, parallel_backend])
def test_backend_hinting_and_constraints_with_custom_backends(capsys, context):

    class MyCustomThreadingBackend(ParallelBackendBase):
        supports_sharedmem = True
        use_threads = True

        def apply_async(self):
            pass

        def effective_n_jobs(self, n_jobs):
            return n_jobs
    with context(MyCustomThreadingBackend()):
        p = Parallel(n_jobs=2, prefer='processes')
        assert type(p._backend) is MyCustomThreadingBackend
        p = Parallel(n_jobs=2, require='sharedmem')
        assert type(p._backend) is MyCustomThreadingBackend

    class MyCustomProcessingBackend(ParallelBackendBase):
        supports_sharedmem = False
        use_threads = False

        def apply_async(self):
            pass

        def effective_n_jobs(self, n_jobs):
            return n_jobs
    with context(MyCustomProcessingBackend()):
        p = Parallel(n_jobs=2, prefer='processes')
        assert type(p._backend) is MyCustomProcessingBackend
        out, err = capsys.readouterr()
        assert out == ''
        assert err == ''
        p = Parallel(n_jobs=2, require='sharedmem', verbose=10)
        assert type(p._backend) is ThreadingBackend
        out, err = capsys.readouterr()
        expected = 'Using ThreadingBackend as joblib backend instead of MyCustomProcessingBackend as the latter does not provide shared memory semantics.'
        assert out.strip() == expected
        assert err == ''
    with raises(ValueError):
        Parallel(backend=MyCustomProcessingBackend(), require='sharedmem')