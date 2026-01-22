from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_chunk_boundaries() -> None:
    conn = Connection(our_role=SERVER)
    request = b'POST / HTTP/1.1\r\nHost: example.com\r\nTransfer-Encoding: chunked\r\n\r\n'
    conn.receive_data(request)
    assert conn.next_event() == Request(method='POST', target='/', headers=[('Host', 'example.com'), ('Transfer-Encoding', 'chunked')])
    assert conn.next_event() is NEED_DATA
    conn.receive_data(b'5\r\nhello\r\n')
    assert conn.next_event() == Data(data=b'hello', chunk_start=True, chunk_end=True)
    conn.receive_data(b'5\r\nhel')
    assert conn.next_event() == Data(data=b'hel', chunk_start=True, chunk_end=False)
    conn.receive_data(b'l')
    assert conn.next_event() == Data(data=b'l', chunk_start=False, chunk_end=False)
    conn.receive_data(b'o\r\n')
    assert conn.next_event() == Data(data=b'o', chunk_start=False, chunk_end=True)
    conn.receive_data(b'5\r\nhello')
    assert conn.next_event() == Data(data=b'hello', chunk_start=True, chunk_end=True)
    conn.receive_data(b'\r\n')
    assert conn.next_event() == NEED_DATA
    conn.receive_data(b'0\r\n\r\n')
    assert conn.next_event() == EndOfMessage()