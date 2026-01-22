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
def test_buffer_from_numpy():
    arr = np.arange(12, dtype=np.int8).reshape((3, 4))
    buf = pa.py_buffer(arr)
    assert buf.is_cpu
    assert buf.is_mutable
    assert buf.to_pybytes() == arr.tobytes()
    buf = pa.py_buffer(arr.T)
    assert buf.is_cpu
    assert buf.is_mutable
    assert buf.to_pybytes() == arr.tobytes()
    with pytest.raises(ValueError, match='not contiguous'):
        buf = pa.py_buffer(arr.T[::2])