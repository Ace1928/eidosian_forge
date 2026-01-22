from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
@pytest.mark.parametrize('data', [b'\x00', b' ', b'\x16\x03\x01\x00\xa5'])
def test_early_detection_of_invalid_request(data: bytes) -> None:
    c = Connection(SERVER)
    c.receive_data(data)
    with pytest.raises(RemoteProtocolError):
        c.next_event()