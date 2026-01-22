import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
def test_buffer_to_numpy():
    byte_array = bytearray(20)
    byte_array[0] = 42
    buf = pa.py_buffer(byte_array)
    array = np.frombuffer(buf, dtype='uint8')
    assert array[0] == byte_array[0]
    byte_array[0] += 1
    assert array[0] == byte_array[0]
    assert array.base == buf