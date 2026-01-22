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
@pytest.mark.parametrize('sink_factory', [lambda: io.BytesIO(), lambda: pa.BufferOutputStream()])
def test_file_read_all(sink_factory):
    fixture = FileFormatFixture(sink_factory)
    batches = fixture.write_batches()
    file_contents = pa.BufferReader(fixture.get_source())
    reader = pa.ipc.open_file(file_contents)
    result = reader.read_all()
    expected = pa.Table.from_batches(batches)
    assert result.equals(expected)