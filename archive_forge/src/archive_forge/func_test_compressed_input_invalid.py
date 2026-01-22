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
@pytest.mark.gzip
def test_compressed_input_invalid():
    data = b'foo' * 10
    raw = pa.BufferReader(data)
    with pytest.raises(ValueError):
        pa.CompressedInputStream(raw, 'unknown_compression')
    with pytest.raises(TypeError):
        pa.CompressedInputStream(raw, None)
    with pa.CompressedInputStream(raw, 'gzip') as compressed:
        with pytest.raises(IOError, match='zlib inflate failed'):
            compressed.read()