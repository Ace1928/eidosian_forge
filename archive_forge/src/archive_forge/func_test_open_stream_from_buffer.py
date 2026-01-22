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
def test_open_stream_from_buffer(stream_fixture):
    stream_fixture.write_batches()
    source = stream_fixture.get_source()
    reader1 = pa.ipc.open_stream(source)
    reader2 = pa.ipc.open_stream(pa.BufferReader(source))
    reader3 = pa.RecordBatchStreamReader(source)
    result1 = reader1.read_all()
    result2 = reader2.read_all()
    result3 = reader3.read_all()
    assert result1.equals(result2)
    assert result1.equals(result3)
    st1 = reader1.stats
    assert st1.num_messages == 6
    assert st1.num_record_batches == 5
    assert reader2.stats == st1
    assert reader3.stats == st1
    assert tuple(st1) == tuple(stream_fixture.write_stats)