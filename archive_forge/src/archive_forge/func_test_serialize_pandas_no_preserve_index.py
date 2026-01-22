from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
@pytest.mark.pandas
def test_serialize_pandas_no_preserve_index():
    df = pd.DataFrame({'a': [1, 2, 3]}, index=[1, 2, 3])
    expected = pd.DataFrame({'a': [1, 2, 3]})
    buf = pa.serialize_pandas(df, preserve_index=False)
    result = pa.deserialize_pandas(buf)
    assert_frame_equal(result, expected)
    buf = pa.serialize_pandas(df, preserve_index=True)
    result = pa.deserialize_pandas(buf)
    assert_frame_equal(result, df)