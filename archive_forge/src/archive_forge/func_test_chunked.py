from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_chunked() -> None:
    p = ConnectionPair()
    p.send(CLIENT, Request(method='GET', target='/', headers=[('Host', 'example.com'), ('Transfer-Encoding', 'chunked')]))
    data = p.send(CLIENT, Data(data=b'1234567890', chunk_start=True, chunk_end=True))
    assert data == b'a\r\n1234567890\r\n'
    data = p.send(CLIENT, Data(data=b'abcde', chunk_start=True, chunk_end=True))
    assert data == b'5\r\nabcde\r\n'
    data = p.send(CLIENT, Data(data=b''), expect=[])
    assert data == b''
    data = p.send(CLIENT, EndOfMessage(headers=[('hello', 'there')]))
    assert data == b'0\r\nhello: there\r\n\r\n'
    p.send(SERVER, Response(status_code=200, headers=[('Transfer-Encoding', 'chunked')]))
    p.send(SERVER, Data(data=b'54321', chunk_start=True, chunk_end=True))
    p.send(SERVER, Data(data=b'12345', chunk_start=True, chunk_end=True))
    p.send(SERVER, EndOfMessage())
    for conn in p.conns:
        assert conn.states == {CLIENT: DONE, SERVER: DONE}