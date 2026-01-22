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
def test_buffered_input_stream_detach_non_seekable():
    f = pa.BufferedInputStream(pa.BufferedInputStream(pa.BufferReader(b'123456789'), buffer_size=4), buffer_size=4)
    assert f.read(2) == b'12'
    raw = f.detach()
    assert f.closed
    assert not raw.closed
    assert not raw.seekable()
    assert raw.read(4) == b'5678'
    with pytest.raises(EnvironmentError):
        raw.seek(2)