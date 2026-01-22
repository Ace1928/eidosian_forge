from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_automagic_connection_close_handling() -> None:
    p = ConnectionPair()
    p.send(CLIENT, [Request(method='GET', target='/', headers=[('Host', 'example.com'), ('Connection', 'close')]), EndOfMessage()])
    for conn in p.conns:
        assert conn.states[CLIENT] is MUST_CLOSE
    p.send(SERVER, [Response(status_code=204, headers=[]), EndOfMessage()], expect=[Response(status_code=204, headers=[('connection', 'close')]), EndOfMessage()])
    for conn in p.conns:
        assert conn.states == {CLIENT: MUST_CLOSE, SERVER: MUST_CLOSE}