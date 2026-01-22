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
@pytest.mark.parametrize('use_legacy_ipc_format', [False, True])
def test_stream_simple_roundtrip(stream_fixture, use_legacy_ipc_format):
    stream_fixture.use_legacy_ipc_format = use_legacy_ipc_format
    batches = stream_fixture.write_batches()
    file_contents = pa.BufferReader(stream_fixture.get_source())
    reader = pa.ipc.open_stream(file_contents)
    assert reader.schema.equals(batches[0].schema)
    total = 0
    for i, next_batch in enumerate(reader):
        assert next_batch.equals(batches[i])
        total += 1
    assert total == len(batches)
    with pytest.raises(StopIteration):
        reader.read_next_batch()