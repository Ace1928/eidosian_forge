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
def test__strided_from_memmap(tmpdir):
    fname = tmpdir.join('test.mmap').strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    memmap_obj = np.memmap(fname, mode='w+', shape=size + offset)
    memmap_obj = _strided_from_memmap(fname, dtype='uint8', mode='r', offset=offset, order='C', shape=size, strides=None, total_buffer_len=None, unlink_on_gc_collect=False)
    assert isinstance(memmap_obj, np.memmap)
    assert memmap_obj.offset == offset
    memmap_backed_obj = _strided_from_memmap(fname, dtype='uint8', mode='r', offset=offset, order='C', shape=(size // 2,), strides=(2,), total_buffer_len=size, unlink_on_gc_collect=False)
    assert _get_backing_memmap(memmap_backed_obj).offset == offset