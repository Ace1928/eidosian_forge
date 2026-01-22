import os
import mmap
import sys
import platform
import gc
import pickle
import itertools
from time import sleep
import subprocess
import threading
import faulthandler
import pytest
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.test.common import with_dev_shm
from joblib.testing import raises, parametrize, skipif
from joblib.backports import make_memmap
from joblib.parallel import Parallel, delayed
from joblib.pool import MemmappingPool
from joblib.executor import _TestingMemmappingExecutor as TestExecutor
from joblib._memmapping_reducer import has_shareable_memory
from joblib._memmapping_reducer import ArrayMemmapForwardReducer
from joblib._memmapping_reducer import _strided_from_memmap
from joblib._memmapping_reducer import _get_temp_dir
from joblib._memmapping_reducer import _WeakArrayKeyMap
from joblib._memmapping_reducer import _get_backing_memmap
import joblib._memmapping_reducer as jmr
def test_pool_get_temp_dir_no_statvfs(tmpdir, monkeypatch):
    """Check that _get_temp_dir works when os.statvfs is not defined

    Regression test for #902
    """
    pool_folder_name = 'test.tmpdir'
    import joblib._memmapping_reducer
    if hasattr(joblib._memmapping_reducer.os, 'statvfs'):
        monkeypatch.delattr(joblib._memmapping_reducer.os, 'statvfs')
    pool_folder, shared_mem = _get_temp_dir(pool_folder_name, temp_folder=None)
    if sys.platform.startswith('win'):
        assert shared_mem is False
    assert pool_folder.endswith(pool_folder_name)