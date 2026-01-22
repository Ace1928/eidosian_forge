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
@parametrize('extension,decompress', [('.z', _zlib_file_decompress), ('.gz', _gzip_file_decompress)])
def test_load_externally_decompressed_files(tmpdir, extension, decompress):
    obj = 'a string to persist'
    filename_raw = tmpdir.join('test.pkl').strpath
    filename_compressed = filename_raw + extension
    numpy_pickle.dump(obj, filename_compressed)
    decompress(filename_compressed, filename_raw)
    obj_reloaded = numpy_pickle.load(filename_raw)
    assert obj == obj_reloaded