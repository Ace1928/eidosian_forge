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
def test_warning_on_unknown_location_type():

    class NonSupportedLocationClass:
        pass
    unsupported_location = NonSupportedLocationClass()
    with warns(UserWarning) as warninfo:
        _store_backend_factory('local', location=unsupported_location)
    expected_mesage = 'Instantiating a backend using a NonSupportedLocationClass as a location is not supported by joblib'
    assert expected_mesage in str(warninfo[0].message)