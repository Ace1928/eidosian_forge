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
def test_memory_map_resize(tmpdir):
    SIZE = 4096
    arr = np.random.randint(0, 256, size=SIZE).astype(np.uint8)
    data1 = arr.tobytes()[:SIZE // 2]
    data2 = arr.tobytes()[SIZE // 2:]
    path = os.path.join(str(tmpdir), guid())
    mmap = pa.create_memory_map(path, SIZE / 2)
    mmap.write(data1)
    mmap.resize(SIZE)
    mmap.write(data2)
    mmap.close()
    with open(path, 'rb') as f:
        assert f.read() == arr.tobytes()