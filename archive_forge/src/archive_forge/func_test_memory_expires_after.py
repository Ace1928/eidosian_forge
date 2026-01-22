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
def test_memory_expires_after(self, memory):
    """Test expiry of old cached results"""
    f = memory.cache(self.foo, cache_validation_callback=expires_after(seconds=0.3), ignore=['d'])
    d1, d2, d3 = ({'run': False}, {'run': False}, {'run': False})
    assert f(2, d1) == 4
    assert f(2, d2) == 4
    time.sleep(0.5)
    assert f(2, d3) == 4
    assert d1['run']
    assert not d2['run']
    assert d3['run']