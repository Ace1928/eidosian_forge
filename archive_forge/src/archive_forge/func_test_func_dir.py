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
def test_func_dir(tmpdir):
    memory = Memory(location=tmpdir.strpath, verbose=0)
    path = __name__.split('.')
    path.append('f')
    path = tmpdir.join('joblib', *path).strpath
    g = memory.cache(f)
    func_id = _build_func_identifier(f)
    location = os.path.join(g.store_backend.location, func_id)
    assert location == path
    assert os.path.exists(path)
    assert memory.location == os.path.dirname(g.store_backend.location)
    _FUNCTION_HASHES.clear()
    assert not g._check_previous_func_code()
    assert os.path.exists(os.path.join(path, 'func_code.py'))
    assert g._check_previous_func_code()
    func_id, args_id = g._get_output_identifiers(1)
    output_dir = os.path.join(g.store_backend.location, func_id, args_id)
    a = g(1)
    assert os.path.exists(output_dir)
    os.remove(os.path.join(output_dir, 'output.pkl'))
    assert a == g(1)