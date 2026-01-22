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
def test_memory_warning_collision_detection(tmpdir):
    memory = Memory(location=tmpdir.strpath, verbose=0)
    a1 = eval('lambda x: x')
    a1 = memory.cache(a1)
    b1 = eval('lambda x: x+1')
    b1 = memory.cache(b1)
    with warns(JobLibCollisionWarning) as warninfo:
        a1(1)
        b1(1)
        a1(0)
    assert len(warninfo) == 2
    assert 'cannot detect' in str(warninfo[0].message).lower()