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
def test_native_file_TextIOWrapper_perf(tmpdir):
    data = b'foo\nquux\n'
    path = str(tmpdir / 'largefile.txt')
    with open(path, 'wb') as f:
        f.write(data * 100000)
    binary_file = pa.OSFile(path, mode='rb')
    with TextIOWrapper(binary_file) as f:
        assert binary_file.tell() == 0
        nbytes = 20000
        lines = f.readlines(nbytes)
        assert len(lines) == math.ceil(2 * nbytes / len(data))
        assert nbytes <= binary_file.tell() <= nbytes * 2