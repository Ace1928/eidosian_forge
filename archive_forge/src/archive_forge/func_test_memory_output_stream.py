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
def test_memory_output_stream():
    val = b'dataabcdef'
    f = pa.BufferOutputStream()
    K = 1000
    for i in range(K):
        f.write(val)
    buf = f.getvalue()
    assert len(buf) == len(val) * K
    assert buf.to_pybytes() == val * K