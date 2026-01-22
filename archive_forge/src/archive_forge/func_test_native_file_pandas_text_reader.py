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
@pytest.mark.pandas
def test_native_file_pandas_text_reader(tmpdir):
    import pandas as pd
    import pandas.testing as tm
    data = b'a,b\n' * 10000000
    path = str(tmpdir / 'largefile.txt')
    with open(path, 'wb') as f:
        f.write(data)
    with pa.OSFile(path, mode='rb') as f:
        df = pd.read_csv(f, nrows=10)
        expected = pd.DataFrame({'a': ['a'] * 10, 'b': ['b'] * 10})
        tm.assert_frame_equal(df, expected)
        assert f.tell() <= 256 * 1024