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
def test_file_handle_persistence_in_memory_mmap():
    obj = np.random.random((10, 10))
    buf = io.BytesIO()
    numpy_pickle.dump(obj, buf)
    with warns(UserWarning) as warninfo:
        numpy_pickle.load(buf, mmap_mode='r+')
    assert len(warninfo) == 1
    assert str(warninfo[0].message) == 'In memory persistence is not compatible with mmap_mode "%(mmap_mode)s" flag passed. mmap_mode option will be ignored.' % {'mmap_mode': 'r+'}