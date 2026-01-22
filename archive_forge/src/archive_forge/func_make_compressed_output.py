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
def make_compressed_output(data, fn, compression):
    raw = pa.BufferOutputStream()
    with pa.CompressedOutputStream(raw, compression) as compressed:
        assert not compressed.closed
        assert not compressed.readable()
        assert compressed.writable()
        assert not compressed.seekable()
        compressed.write(data)
    assert compressed.closed
    assert raw.closed
    with open(fn, 'wb') as f:
        f.write(raw.getvalue())