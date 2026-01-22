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
@parametrize('factory,retry_no', list(itertools.product([MemmappingPool, TestExecutor.get_memmapping_executor], range(3))), ids=['{}, {}'.format(x, y) for x, y in itertools.product(['multiprocessing', 'loky'], map(str, range(3)))])
def test_pool_memmap_with_big_offset(factory, retry_no, tmpdir):
    fname = tmpdir.join('test.mmap').strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    obj = make_memmap(fname, mode='w+', shape=size, dtype='uint8', offset=offset)
    p = factory(2, temp_folder=tmpdir.strpath)
    result = p.apply_async(identity, args=(obj,)).get()
    assert isinstance(result, np.memmap)
    assert result.offset == offset
    np.testing.assert_array_equal(obj, result)
    p.terminate()