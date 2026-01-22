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
def test_python_file_iterable():
    data = b'line1\n    line2\n    line3\n    '
    buf = BytesIO(data)
    buf2 = BytesIO(data)
    with pa.PythonFile(buf, mode='r') as f:
        for read, expected in zip(f, buf2):
            assert read == expected