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
def test_compress_mmap_mode_warning(tmpdir):
    rnd = np.random.RandomState(0)
    a = rnd.random_sample(10)
    this_filename = tmpdir.join('test.pkl').strpath
    numpy_pickle.dump(a, this_filename, compress=1)
    with warns(UserWarning) as warninfo:
        numpy_pickle.load(this_filename, mmap_mode='r+')
    debug_msg = '\n'.join([str(w) for w in warninfo])
    warninfo = [w.message for w in warninfo]
    assert len(warninfo) == 1, debug_msg
    assert str(warninfo[0]) == f'mmap_mode "r+" is not compatible with compressed file {this_filename}. "r+" flag will be ignored.'