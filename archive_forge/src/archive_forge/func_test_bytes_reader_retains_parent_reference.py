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
def test_bytes_reader_retains_parent_reference():
    import gc

    def get_buffer():
        data = b'some sample data' * 1000
        reader = pa.BufferReader(data)
        reader.seek(5)
        return reader.read_buffer(6)
    buf = get_buffer()
    gc.collect()
    assert buf.to_pybytes() == b'sample'
    assert buf.parent is not None