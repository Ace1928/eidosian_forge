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
def test_numpy_array_byte_order_mismatch_detection():
    be_arrays = [np.array([(1, 2.0), (3, 4.0)], dtype=[('', '>i8'), ('', '>f8')]), np.arange(3, dtype=np.dtype('>i8')), np.arange(3, dtype=np.dtype('>f8'))]
    for array in be_arrays:
        if sys.byteorder == 'big':
            assert not _is_numpy_array_byte_order_mismatch(array)
        else:
            assert _is_numpy_array_byte_order_mismatch(array)
        converted = _ensure_native_byte_order(array)
        if converted.dtype.fields:
            for f in converted.dtype.fields.values():
                f[0].byteorder == '='
        else:
            assert converted.dtype.byteorder == '='
    le_arrays = [np.array([(1, 2.0), (3, 4.0)], dtype=[('', '<i8'), ('', '<f8')]), np.arange(3, dtype=np.dtype('<i8')), np.arange(3, dtype=np.dtype('<f8'))]
    for array in le_arrays:
        if sys.byteorder == 'little':
            assert not _is_numpy_array_byte_order_mismatch(array)
        else:
            assert _is_numpy_array_byte_order_mismatch(array)
        converted = _ensure_native_byte_order(array)
        if converted.dtype.fields:
            for f in converted.dtype.fields.values():
                f[0].byteorder == '='
        else:
            assert converted.dtype.byteorder == '='