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
def test_python_file_write():
    buf = BytesIO()
    f = pa.PythonFile(buf)
    assert f.tell() == 0
    s1 = b'enga\xc3\xb1ado'
    s2 = b'foobar'
    f.write(s1)
    assert f.tell() == len(s1)
    f.write(s2)
    expected = s1 + s2
    result = buf.getvalue()
    assert result == expected
    assert not f.closed
    f.close()
    assert f.closed
    with pytest.raises(TypeError, match='binary file expected'):
        pa.PythonFile(StringIO())