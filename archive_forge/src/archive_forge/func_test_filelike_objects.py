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
@pytest.mark.pandas
def test_filelike_objects(version):
    buf = io.BytesIO()
    df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=['a', 'b', 'c']).copy()
    write_feather(df, buf, version=version)
    buf.seek(0)
    result = read_feather(buf)
    assert_frame_equal(result, df)