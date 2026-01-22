from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_pipelined_close() -> None:
    c = Connection(SERVER)
    c.receive_data(b'GET /1 HTTP/1.1\r\nHost: a.com\r\nContent-Length: 5\r\n\r\n12345GET /2 HTTP/1.1\r\nHost: a.com\r\nContent-Length: 5\r\n\r\n67890')
    c.receive_data(b'')
    assert get_all_events(c) == [Request(method='GET', target='/1', headers=[('host', 'a.com'), ('content-length', '5')]), Data(data=b'12345'), EndOfMessage()]
    assert c.states[CLIENT] is DONE
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    assert c.states[SERVER] is DONE
    c.start_next_cycle()
    assert get_all_events(c) == [Request(method='GET', target='/2', headers=[('host', 'a.com'), ('content-length', '5')]), Data(data=b'67890'), EndOfMessage(), ConnectionClosed()]
    assert c.states == {CLIENT: CLOSED, SERVER: SEND_RESPONSE}
    c.send(Response(status_code=200, headers=[]))
    c.send(EndOfMessage())
    assert c.states == {CLIENT: CLOSED, SERVER: MUST_CLOSE}
    c.send(ConnectionClosed())
    assert c.states == {CLIENT: CLOSED, SERVER: CLOSED}