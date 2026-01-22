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
def test_allocate_buffer():
    buf = pa.allocate_buffer(100)
    assert buf.size == 100
    assert buf.is_mutable
    assert buf.parent is None
    bit = b'abcde'
    writer = pa.FixedSizeBufferWriter(buf)
    writer.write(bit)
    assert buf.to_pybytes()[:5] == bit