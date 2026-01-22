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
def test_stream_read_all(stream_fixture):
    batches = stream_fixture.write_batches()
    file_contents = pa.BufferReader(stream_fixture.get_source())
    reader = pa.ipc.open_stream(file_contents)
    result = reader.read_all()
    expected = pa.Table.from_batches(batches)
    assert result.equals(expected)