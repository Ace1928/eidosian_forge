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
def test_non_contiguous_array_pickling(tmpdir):
    filename = tmpdir.join('test.pkl').strpath
    for array in [np.asfortranarray([[1, 2], [3, 4]])[1:], np.ones((10, 50, 20), order='F')[:, :1, :]]:
        assert not array.flags.c_contiguous
        assert not array.flags.f_contiguous
        numpy_pickle.dump(array, filename)
        array_reloaded = numpy_pickle.load(filename)
        np.testing.assert_array_equal(array_reloaded, array)