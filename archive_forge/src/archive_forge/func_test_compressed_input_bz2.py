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
def test_compressed_input_bz2(tmpdir):
    data = b'some test data\n' * 10 + b'eof\n'
    fn = str(tmpdir / 'compressed_input_test.bz2')
    with bz2.BZ2File(fn, 'w') as f:
        f.write(data)
    try:
        check_compressed_input(data, fn, 'bz2')
    except NotImplementedError as e:
        pytest.skip(str(e))