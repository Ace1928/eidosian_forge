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
def test_file_manager_acquire_context(tmpdir, file_cache) -> None:
    path = str(tmpdir.join('testing.txt'))
    with open(path, 'w') as f:
        f.write('foobar')

    class AcquisitionError(Exception):
        pass
    manager = CachingFileManager(open, path, cache=file_cache)
    with pytest.raises(AcquisitionError):
        with manager.acquire_context() as f:
            assert f.read() == 'foobar'
            raise AcquisitionError
    assert not file_cache
    with manager.acquire_context() as f:
        assert f.read() == 'foobar'
    with pytest.raises(AcquisitionError):
        with manager.acquire_context() as f:
            f.seek(0)
            assert f.read() == 'foobar'
            raise AcquisitionError
    assert file_cache
    manager.close()