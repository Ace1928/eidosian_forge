import os
import shutil
import time
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest
from ..utils import (
@pytest.mark.parametrize('pool', [ThreadPoolExecutor, ProcessPoolExecutor], ids=['threads', 'processes'])
def test_make_local_storage_parallel(pool, monkeypatch):
    """Try to create the cache folder in parallel"""
    makedirs = os.makedirs

    def mockmakedirs(path, exist_ok=False):
        """Delay before calling makedirs"""
        time.sleep(1.5)
        makedirs(path, exist_ok=exist_ok)
    monkeypatch.setattr(os, 'makedirs', mockmakedirs)
    data_cache = os.path.join(os.curdir, 'test_parallel_cache')
    assert not os.path.exists(data_cache)
    try:
        with pool() as executor:
            futures = [executor.submit(make_local_storage, data_cache) for i in range(4)]
            for future in futures:
                future.result()
            assert os.path.exists(data_cache)
    finally:
        if os.path.exists(data_cache):
            shutil.rmtree(data_cache)