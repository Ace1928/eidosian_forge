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
@pytest.mark.parametrize('compression', ['bz2', 'brotli', 'gzip', 'lz4', 'zstd'])
def test_compressed_recordbatch_stream(compression):
    if not Codec.is_available(compression):
        pytest.skip('{} support is not built'.format(compression))
    table = pa.Table.from_arrays([pa.array([1, 2, 3, 4, 5])], ['a'])
    raw = pa.BufferOutputStream()
    stream = pa.CompressedOutputStream(raw, compression)
    writer = pa.RecordBatchStreamWriter(stream, table.schema)
    writer.write_table(table, max_chunksize=3)
    writer.close()
    stream.close()
    buf = raw.getvalue()
    stream = pa.CompressedInputStream(pa.BufferReader(buf), compression)
    got_table = pa.RecordBatchStreamReader(stream).read_all()
    assert got_table == table