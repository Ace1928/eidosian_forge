from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_client_talking_to_http10_server() -> None:
    c = Connection(CLIENT)
    c.send(Request(method='GET', target='/', headers=[('Host', 'example.com')]))
    c.send(EndOfMessage())
    assert c.our_state is DONE
    assert receive_and_get(c, b'HTTP/1.0 200 OK\r\n\r\n') == [Response(status_code=200, headers=[], http_version='1.0', reason=b'OK')]
    assert c.our_state is MUST_CLOSE
    assert receive_and_get(c, b'12345') == [Data(data=b'12345')]
    assert receive_and_get(c, b'67890') == [Data(data=b'67890')]
    assert receive_and_get(c, b'') == [EndOfMessage(), ConnectionClosed()]
    assert c.their_state is CLOSED