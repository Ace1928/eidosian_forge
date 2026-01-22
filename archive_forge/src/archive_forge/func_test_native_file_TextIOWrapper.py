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
def test_native_file_TextIOWrapper(tmpdir):
    data = 'foooo\nbarrr\nbazzz\n'
    path = os.path.join(str(tmpdir), guid())
    with open(path, 'wb') as f:
        f.write(data.encode('utf-8'))
    with TextIOWrapper(pa.OSFile(path, mode='rb')) as fil:
        assert fil.readable()
        res = fil.read()
        assert res == data
    assert fil.closed
    with TextIOWrapper(pa.OSFile(path, mode='rb')) as fil:
        lines = list(fil)
        assert ''.join(lines) == data
    path2 = os.path.join(str(tmpdir), guid())
    with TextIOWrapper(pa.OSFile(path2, mode='wb')) as fil:
        assert fil.writable()
        fil.write(data)
    with TextIOWrapper(pa.OSFile(path2, mode='rb')) as fil:
        res = fil.read()
        assert res == data