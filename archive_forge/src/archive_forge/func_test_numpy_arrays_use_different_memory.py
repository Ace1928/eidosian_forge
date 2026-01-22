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
@with_numpy
@skipif(sys.platform == 'win32', reason='This test fails with a PermissionError on Windows')
@parametrize('mmap_mode', ['r+', 'w+'])
def test_numpy_arrays_use_different_memory(mmap_mode):

    def func(arr, value):
        arr[:] = value
        return arr
    arrays = [np.zeros((10, 10), dtype='float64') for i in range(10)]
    results = Parallel(mmap_mode=mmap_mode, max_nbytes=0, n_jobs=2)((delayed(func)(arr, i) for i, arr in enumerate(arrays)))
    for i, arr in enumerate(results):
        np.testing.assert_array_equal(arr, i)