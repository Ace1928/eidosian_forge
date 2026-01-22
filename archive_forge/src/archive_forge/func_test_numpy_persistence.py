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
@parametrize('compress', [False, True, 0, 3, 'zlib'])
def test_numpy_persistence(tmpdir, compress):
    filename = tmpdir.join('test.pkl').strpath
    rnd = np.random.RandomState(0)
    a = rnd.random_sample((10, 2))
    for index, obj in enumerate(((a,), (a.T,), (a, a), [a, a, a])):
        filenames = numpy_pickle.dump(obj, filename, compress=compress)
        assert len(filenames) == 1
        assert filenames[0] == filename
        assert os.path.exists(filenames[0])
        obj_ = numpy_pickle.load(filename)
        for item in obj_:
            assert isinstance(item, np.ndarray)
        np.testing.assert_array_equal(np.array(obj), np.array(obj_))
    obj = np.memmap(filename + 'mmap', mode='w+', shape=4, dtype=np.float64)
    filenames = numpy_pickle.dump(obj, filename, compress=compress)
    assert len(filenames) == 1
    obj_ = numpy_pickle.load(filename)
    if type(obj) is not np.memmap and hasattr(obj, '__array_prepare__'):
        assert isinstance(obj_, type(obj))
    np.testing.assert_array_equal(obj_, obj)
    obj = ComplexTestObject()
    filenames = numpy_pickle.dump(obj, filename, compress=compress)
    assert len(filenames) == 1
    obj_loaded = numpy_pickle.load(filename)
    assert isinstance(obj_loaded, type(obj))
    np.testing.assert_array_equal(obj_loaded.array_float, obj.array_float)
    np.testing.assert_array_equal(obj_loaded.array_int, obj.array_int)
    np.testing.assert_array_equal(obj_loaded.array_obj, obj.array_obj)