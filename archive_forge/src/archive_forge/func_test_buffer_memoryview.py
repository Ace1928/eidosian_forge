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
def test_buffer_memoryview(pickle_module):
    val = b'some data'
    buf = pa.py_buffer(val)
    assert isinstance(buf, pa.Buffer)
    assert not buf.is_mutable
    assert buf.is_cpu
    result = memoryview(buf)
    assert result == val
    check_buffer_pickling(buf, pickle_module)