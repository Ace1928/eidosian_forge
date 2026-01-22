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
def test_call_and_shelve(tmpdir):
    for func, Result in zip((MemorizedFunc(f, tmpdir.strpath), NotMemorizedFunc(f), Memory(location=tmpdir.strpath, verbose=0).cache(f), Memory(location=None).cache(f)), (MemorizedResult, NotMemorizedResult, MemorizedResult, NotMemorizedResult)):
        assert func(2) == 5
        result = func.call_and_shelve(2)
        assert isinstance(result, Result)
        assert result.get() == 5
        result.clear()
        with raises(KeyError):
            result.get()
        result.clear()