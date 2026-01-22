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
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_parallel_isolated_temp_folders(backend):
    array = np.arange(int(100.0))
    [filename_1] = Parallel(n_jobs=2, backend=backend, max_nbytes=10)((delayed(getattr)(array, 'filename') for _ in range(1)))
    [filename_2] = Parallel(n_jobs=2, backend=backend, max_nbytes=10)((delayed(getattr)(array, 'filename') for _ in range(1)))
    assert os.path.dirname(filename_2) != os.path.dirname(filename_1)