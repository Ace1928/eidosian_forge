from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_Connection_basics_and_content_length() -> None:
    with pytest.raises(ValueError):
        Connection('CLIENT')
    p = ConnectionPair()
    assert p.conn[CLIENT].our_role is CLIENT
    assert p.conn[CLIENT].their_role is SERVER
    assert p.conn[SERVER].our_role is SERVER
    assert p.conn[SERVER].their_role is CLIENT
    data = p.send(CLIENT, Request(method='GET', target='/', headers=[('Host', 'example.com'), ('Content-Length', '10')]))
    assert data == b'GET / HTTP/1.1\r\nHost: example.com\r\nContent-Length: 10\r\n\r\n'
    for conn in p.conns:
        assert conn.states == {CLIENT: SEND_BODY, SERVER: SEND_RESPONSE}
    assert p.conn[CLIENT].our_state is SEND_BODY
    assert p.conn[CLIENT].their_state is SEND_RESPONSE
    assert p.conn[SERVER].our_state is SEND_RESPONSE
    assert p.conn[SERVER].their_state is SEND_BODY
    assert p.conn[CLIENT].their_http_version is None
    assert p.conn[SERVER].their_http_version == b'1.1'
    data = p.send(SERVER, InformationalResponse(status_code=100, headers=[]))
    assert data == b'HTTP/1.1 100 \r\n\r\n'
    data = p.send(SERVER, Response(status_code=200, headers=[('Content-Length', '11')]))
    assert data == b'HTTP/1.1 200 \r\nContent-Length: 11\r\n\r\n'
    for conn in p.conns:
        assert conn.states == {CLIENT: SEND_BODY, SERVER: SEND_BODY}
    assert p.conn[CLIENT].their_http_version == b'1.1'
    assert p.conn[SERVER].their_http_version == b'1.1'
    data = p.send(CLIENT, Data(data=b'12345'))
    assert data == b'12345'
    data = p.send(CLIENT, Data(data=b'67890'), expect=[Data(data=b'67890'), EndOfMessage()])
    assert data == b'67890'
    data = p.send(CLIENT, EndOfMessage(), expect=[])
    assert data == b''
    for conn in p.conns:
        assert conn.states == {CLIENT: DONE, SERVER: SEND_BODY}
    data = p.send(SERVER, Data(data=b'1234567890'))
    assert data == b'1234567890'
    data = p.send(SERVER, Data(data=b'1'), expect=[Data(data=b'1'), EndOfMessage()])
    assert data == b'1'
    data = p.send(SERVER, EndOfMessage(), expect=[])
    assert data == b''
    for conn in p.conns:
        assert conn.states == {CLIENT: DONE, SERVER: DONE}