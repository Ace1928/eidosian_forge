from __future__ import annotations
import gc
import pickle
import threading
from unittest import mock
import pytest
from xarray.backends.file_manager import CachingFileManager
from xarray.backends.lru_cache import LRUCache
from xarray.core.options import set_options
from xarray.tests import assert_no_warnings
def test_file_manager_write_pickle(tmpdir, file_cache) -> None:
    path = str(tmpdir.join('testing.txt'))
    manager = CachingFileManager(open, path, mode='w', cache=file_cache)
    f = manager.acquire()
    f.write('foo')
    f.flush()
    manager2 = pickle.loads(pickle.dumps(manager))
    f2 = manager2.acquire()
    f2.write('bar')
    manager2.close()
    manager.close()
    with open(path) as f:
        assert f.read() == 'foobar'