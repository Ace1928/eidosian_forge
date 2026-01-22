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
def test_memory_file_modification(capsys, tmpdir, monkeypatch):
    dir_name = tmpdir.mkdir('tmp_import').strpath
    filename = os.path.join(dir_name, 'tmp_joblib_.py')
    content = 'def f(x):\n    print(x)\n    return x\n'
    with open(filename, 'w') as module_file:
        module_file.write(content)
    monkeypatch.syspath_prepend(dir_name)
    import tmp_joblib_ as tmp
    memory = Memory(location=tmpdir.strpath, verbose=0)
    f = memory.cache(tmp.f)
    f(1)
    f(2)
    f(1)
    with open(filename, 'w') as module_file:
        module_file.write('\n\n' + content)
    f(1)
    f(1)
    shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    content = 'def f(x):\n    print("x=%s" % x)\n    return x\n'
    with open(filename, 'w') as module_file:
        module_file.write(content)
    f(1)
    f(1)
    sys.stdout.write('Reloading\n')
    sys.modules.pop('tmp_joblib_')
    import tmp_joblib_ as tmp
    f = memory.cache(tmp.f)
    f(1)
    f(1)
    out, err = capsys.readouterr()
    assert out == '1\n2\nReloading\nx=1\n'