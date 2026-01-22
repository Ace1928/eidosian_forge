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
def test_message_serialize_read_message(example_messages):
    _, messages = example_messages
    msg = messages[0]
    buf = msg.serialize()
    reader = pa.BufferReader(buf.to_pybytes() * 2)
    restored = pa.ipc.read_message(buf)
    restored2 = pa.ipc.read_message(reader)
    restored3 = pa.ipc.read_message(buf.to_pybytes())
    restored4 = pa.ipc.read_message(reader)
    assert msg.equals(restored)
    assert msg.equals(restored2)
    assert msg.equals(restored3)
    assert msg.equals(restored4)
    with pytest.raises(pa.ArrowInvalid, match='Corrupted message'):
        pa.ipc.read_message(pa.BufferReader(b'ab'))
    with pytest.raises(EOFError):
        pa.ipc.read_message(reader)