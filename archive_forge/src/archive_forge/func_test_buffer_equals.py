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
def test_buffer_equals():

    def eq(a, b):
        assert a.equals(b)
        assert a == b
        assert not a != b

    def ne(a, b):
        assert not a.equals(b)
        assert not a == b
        assert a != b
    b1 = b'some data!'
    b2 = bytearray(b1)
    b3 = bytearray(b1)
    b3[0] = 42
    buf1 = pa.py_buffer(b1)
    buf2 = pa.py_buffer(b2)
    buf3 = pa.py_buffer(b2)
    buf4 = pa.py_buffer(b3)
    buf5 = pa.py_buffer(np.frombuffer(b2, dtype=np.int16))
    eq(buf1, buf1)
    eq(buf1, buf2)
    eq(buf2, buf3)
    ne(buf2, buf4)
    eq(buf2, buf5)