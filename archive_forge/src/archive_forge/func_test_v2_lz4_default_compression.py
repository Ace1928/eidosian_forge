import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
def test_v2_lz4_default_compression():
    if not pa.Codec.is_available('lz4_frame'):
        pytest.skip('LZ4 compression support is not built in C++')
    t = pa.table([np.repeat(0, 100000)], names=['f0'])
    buf = io.BytesIO()
    write_feather(t, buf)
    default_result = buf.getvalue()
    buf = io.BytesIO()
    write_feather(t, buf, compression='uncompressed')
    uncompressed_result = buf.getvalue()
    assert len(default_result) < len(uncompressed_result)