import functools
import gc
import logging
import shutil
import os
import os.path
import pathlib
import pickle
import sys
import time
import datetime
import textwrap
import pytest
from joblib.memory import Memory
from joblib.memory import expires_after
from joblib.memory import MemorizedFunc, NotMemorizedFunc
from joblib.memory import MemorizedResult, NotMemorizedResult
from joblib.memory import _FUNCTION_HASHES
from joblib.memory import register_store_backend, _STORE_BACKENDS
from joblib.memory import _build_func_identifier, _store_backend_factory
from joblib.memory import JobLibCollisionWarning
from joblib.parallel import Parallel, delayed
from joblib._store_backends import StoreBackendBase, FileSystemStoreBackend
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, warns
from joblib.hashing import hash
@with_numpy
@parametrize('mmap_mode', [None, 'r'])
def test_memory_numpy(tmpdir, mmap_mode):
    """ Test memory with a function with numpy arrays."""
    accumulator = list()

    def n(arg=None):
        accumulator.append(1)
        return arg
    memory = Memory(location=tmpdir.strpath, mmap_mode=mmap_mode, verbose=0)
    cached_n = memory.cache(n)
    rnd = np.random.RandomState(0)
    for i in range(3):
        a = rnd.random_sample((10, 10))
        for _ in range(3):
            assert np.all(cached_n(a) == a)
            assert len(accumulator) == i + 1