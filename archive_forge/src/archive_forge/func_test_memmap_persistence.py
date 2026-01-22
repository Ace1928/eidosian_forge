import copy
import os
import random
import re
import io
import sys
import warnings
import gzip
import zlib
import bz2
import pickle
import socket
from contextlib import closing
import mmap
from pathlib import Path
import pytest
from joblib.test.common import np, with_numpy, with_lz4, without_lz4
from joblib.test.common import with_memory_profiler, memory_used
from joblib.testing import parametrize, raises, warns
from joblib import numpy_pickle, register_compressor
from joblib.test import data
from joblib.numpy_pickle_utils import _IO_BUFFER_SIZE
from joblib.numpy_pickle_utils import _detect_compressor
from joblib.numpy_pickle_utils import _is_numpy_array_byte_order_mismatch
from joblib.numpy_pickle_utils import _ensure_native_byte_order
from joblib.compressor import (_COMPRESSORS, _LZ4_PREFIX, CompressorWrapper,
@with_numpy
def test_memmap_persistence(tmpdir):
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    filename = tmpdir.join('test1.pkl').strpath
    numpy_pickle.dump(a, filename)
    b = numpy_pickle.load(filename, mmap_mode='r')
    assert isinstance(b, np.memmap)
    filename = tmpdir.join('test2.pkl').strpath
    obj = ComplexTestObject()
    numpy_pickle.dump(obj, filename)
    obj_loaded = numpy_pickle.load(filename, mmap_mode='r')
    assert isinstance(obj_loaded, type(obj))
    assert isinstance(obj_loaded.array_float, np.memmap)
    assert not obj_loaded.array_float.flags.writeable
    assert isinstance(obj_loaded.array_int, np.memmap)
    assert not obj_loaded.array_int.flags.writeable
    assert not isinstance(obj_loaded.array_obj, np.memmap)
    np.testing.assert_array_equal(obj_loaded.array_float, obj.array_float)
    np.testing.assert_array_equal(obj_loaded.array_int, obj.array_int)
    np.testing.assert_array_equal(obj_loaded.array_obj, obj.array_obj)
    obj_loaded = numpy_pickle.load(filename, mmap_mode='r+')
    assert obj_loaded.array_float.flags.writeable
    obj_loaded.array_float[0:10] = 10.0
    assert obj_loaded.array_int.flags.writeable
    obj_loaded.array_int[0:10] = 10
    obj_reloaded = numpy_pickle.load(filename, mmap_mode='r')
    np.testing.assert_array_equal(obj_reloaded.array_float, obj_loaded.array_float)
    np.testing.assert_array_equal(obj_reloaded.array_int, obj_loaded.array_int)
    numpy_pickle.load(filename, mmap_mode='w+')
    assert obj_loaded.array_int.flags.writeable
    assert obj_loaded.array_int.mode == 'r+'
    assert obj_loaded.array_float.flags.writeable
    assert obj_loaded.array_float.mode == 'r+'