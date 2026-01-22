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
def test_file_manager_write_concurrent(tmpdir, file_cache) -> None:
    path = str(tmpdir.join('testing.txt'))
    manager = CachingFileManager(open, path, mode='w', cache=file_cache)
    f1 = manager.acquire()
    f2 = manager.acquire()
    f3 = manager.acquire()
    assert f1 is f2
    assert f2 is f3
    f1.write('foo')
    f1.flush()
    f2.write('bar')
    f2.flush()
    f3.write('baz')
    f3.flush()
    manager.close()
    with open(path) as f:
        assert f.read() == 'foobarbaz'