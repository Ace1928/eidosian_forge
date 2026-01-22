from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_100_continue() -> None:

    def setup() -> ConnectionPair:
        p = ConnectionPair()
        p.send(CLIENT, Request(method='GET', target='/', headers=[('Host', 'example.com'), ('Content-Length', '100'), ('Expect', '100-continue')]))
        for conn in p.conns:
            assert conn.client_is_waiting_for_100_continue
        assert not p.conn[CLIENT].they_are_waiting_for_100_continue
        assert p.conn[SERVER].they_are_waiting_for_100_continue
        return p
    p = setup()
    p.send(SERVER, InformationalResponse(status_code=100, headers=[]))
    for conn in p.conns:
        assert not conn.client_is_waiting_for_100_continue
        assert not conn.they_are_waiting_for_100_continue
    p = setup()
    p.send(SERVER, Response(status_code=200, headers=[('Transfer-Encoding', 'chunked')]))
    for conn in p.conns:
        assert not conn.client_is_waiting_for_100_continue
        assert not conn.they_are_waiting_for_100_continue
    p = setup()
    p.send(CLIENT, Data(data=b'12345'))
    for conn in p.conns:
        assert not conn.client_is_waiting_for_100_continue
        assert not conn.they_are_waiting_for_100_continue