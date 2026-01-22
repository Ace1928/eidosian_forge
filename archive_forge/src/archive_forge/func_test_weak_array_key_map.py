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
def test_weak_array_key_map():

    def assert_empty_after_gc_collect(container, retries=100):
        for i in range(retries):
            if len(container) == 0:
                return
            gc.collect()
            sleep(0.1)
        assert len(container) == 0
    a = np.ones(42)
    m = _WeakArrayKeyMap()
    m.set(a, 'a')
    assert m.get(a) == 'a'
    b = a
    assert m.get(b) == 'a'
    m.set(b, 'b')
    assert m.get(a) == 'b'
    del a
    gc.collect()
    assert len(m._data) == 1
    assert m.get(b) == 'b'
    del b
    assert_empty_after_gc_collect(m._data)
    c = np.ones(42)
    m.set(c, 'c')
    assert len(m._data) == 1
    assert m.get(c) == 'c'
    with raises(KeyError):
        m.get(np.ones(42))
    del c
    assert_empty_after_gc_collect(m._data)

    def get_set_get_collect(m, i):
        a = np.ones(42)
        with raises(KeyError):
            m.get(a)
        m.set(a, i)
        assert m.get(a) == i
        return id(a)
    unique_ids = set([get_set_get_collect(m, i) for i in range(1000)])
    if platform.python_implementation() == 'CPython':
        max_len_unique_ids = 400 if getattr(sys.flags, 'nogil', False) else 100
        assert len(unique_ids) < max_len_unique_ids