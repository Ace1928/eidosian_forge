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
def test_memory_kwarg(tmpdir):
    """ Test memory with a function with keyword arguments."""
    accumulator = list()

    def g(arg1=None, arg2=1):
        accumulator.append(1)
        return arg1
    check_identity_lazy(g, accumulator, tmpdir.strpath)
    memory = Memory(location=tmpdir.strpath, verbose=0)
    g = memory.cache(g)
    assert g(arg1=30, arg2=2) == 30