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
@with_multiprocessing
def test_cached_function_race_condition_when_persisting_output(tmpdir, capfd):
    memory = Memory(location=tmpdir.strpath)
    func_cached = memory.cache(fast_func_with_complex_output)
    Parallel(n_jobs=2)((delayed(func_cached)() for i in range(3)))
    stdout, stderr = capfd.readouterr()
    exception_msg = 'Exception while loading results'
    assert exception_msg not in stdout
    assert exception_msg not in stderr