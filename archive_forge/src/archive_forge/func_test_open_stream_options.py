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
@pytest.mark.parametrize('options', [pa.ipc.IpcReadOptions(), pa.ipc.IpcReadOptions(use_threads=False)])
def test_open_stream_options(stream_fixture, options):
    stream_fixture.write_batches()
    source = stream_fixture.get_source()
    reader = pa.ipc.open_stream(source, options=options)
    reader.read_all()
    st = reader.stats
    assert st.num_messages == 6
    assert st.num_record_batches == 5
    assert tuple(st) == tuple(stream_fixture.write_stats)