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
def test_socket_read_all(socket_fixture):
    socket_fixture.start_server(do_read_all=True)
    writer_batches = socket_fixture.write_batches()
    _, result = socket_fixture.stop_and_get_result()
    expected = pa.Table.from_batches(writer_batches)
    assert result.equals(expected)