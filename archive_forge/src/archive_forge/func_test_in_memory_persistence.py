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
def test_in_memory_persistence():
    objs = [np.random.random((10, 10)), 'some data']
    for obj in objs:
        f = io.BytesIO()
        numpy_pickle.dump(obj, f)
        obj_reloaded = numpy_pickle.load(f)
        if isinstance(obj, np.ndarray):
            np.testing.assert_array_equal(obj_reloaded, obj)
        else:
            assert obj_reloaded == obj