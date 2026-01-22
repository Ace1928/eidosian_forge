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
def test_read_record_batch_on_stream_error_message():
    batch = pa.record_batch([pa.array([b'foo'], type=pa.utf8())], names=['strs'])
    stream = pa.BufferOutputStream()
    with pa.ipc.new_stream(stream, batch.schema) as writer:
        writer.write_batch(batch)
    buf = stream.getvalue()
    with pytest.raises(IOError, match='type record batch but got schema'):
        pa.ipc.read_record_batch(buf, batch.schema)