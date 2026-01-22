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
def test_file_manager_cache_with_pickle(tmpdir) -> None:
    path = str(tmpdir.join('testing.txt'))
    with open(path, 'w') as f:
        f.write('data')
    cache: dict = {}
    with mock.patch('xarray.backends.file_manager.FILE_CACHE', cache):
        assert not cache
        manager = CachingFileManager(open, path, mode='r')
        manager.acquire()
        assert len(cache) == 1
        manager2 = pickle.loads(pickle.dumps(manager))
        manager2.acquire()
        assert len(cache) == 1
        with set_options(warn_for_unclosed_files=False):
            del manager
            gc.collect()
        with set_options(warn_for_unclosed_files=False):
            del manager2
            gc.collect()
        assert not cache