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
def check_large_seeks(file_factory):
    if sys.platform in ('win32', 'darwin'):
        pytest.skip('need sparse file support')
    try:
        filename = tempfile.mktemp(prefix='test_io')
        with open(filename, 'wb') as f:
            f.truncate(2 ** 32 + 10)
            f.seek(2 ** 32 + 5)
            f.write(b'mark\n')
        with file_factory(filename) as f:
            assert f.seek(2 ** 32 + 5) == 2 ** 32 + 5
            assert f.tell() == 2 ** 32 + 5
            assert f.read(5) == b'mark\n'
            assert f.tell() == 2 ** 32 + 10
    finally:
        os.unlink(filename)