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
def test_ipc_stream_no_batches():
    table = pa.Table.from_arrays([pa.array([1, 2, 3, 4]), pa.array(['foo', 'bar', 'baz', 'qux'])], names=['a', 'b'])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema):
        pass
    source = sink.getvalue()
    with pa.ipc.open_stream(source) as reader:
        result = reader.read_all()
    assert result.schema.equals(table.schema)
    assert len(result) == 0