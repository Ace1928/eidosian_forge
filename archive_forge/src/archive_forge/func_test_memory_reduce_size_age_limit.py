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
def test_memory_reduce_size_age_limit(tmpdir):
    import time
    import datetime
    memory, _, put_cache = _setup_toy_cache(tmpdir)
    ref_cache_items = memory.store_backend.get_items()
    memory.reduce_size()
    cache_items = memory.store_backend.get_items()
    assert sorted(ref_cache_items) == sorted(cache_items)
    memory.reduce_size(age_limit=datetime.timedelta(days=1))
    cache_items = memory.store_backend.get_items()
    assert sorted(ref_cache_items) == sorted(cache_items)
    time.sleep(1)
    put_cache(-1)
    put_cache(-2)
    memory.reduce_size(age_limit=datetime.timedelta(seconds=1))
    cache_items = memory.store_backend.get_items()
    assert not set.issubset(set(cache_items), set(ref_cache_items))
    assert len(cache_items) == 2
    memory.reduce_size(age_limit=datetime.timedelta(seconds=0))
    cache_items = memory.store_backend.get_items()
    assert cache_items == []