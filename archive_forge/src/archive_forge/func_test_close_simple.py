from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test_close_simple() -> None:
    for who_shot_first, who_shot_second in [(CLIENT, SERVER), (SERVER, CLIENT)]:

        def setup() -> ConnectionPair:
            p = ConnectionPair()
            p.send(who_shot_first, ConnectionClosed())
            for conn in p.conns:
                assert conn.states == {who_shot_first: CLOSED, who_shot_second: MUST_CLOSE}
            return p
        p = setup()
        assert p.conn[who_shot_second].next_event() == ConnectionClosed()
        assert p.conn[who_shot_second].next_event() == ConnectionClosed()
        p.conn[who_shot_second].receive_data(b'')
        assert p.conn[who_shot_second].next_event() == ConnectionClosed()
        p = setup()
        p.send(who_shot_second, ConnectionClosed())
        for conn in p.conns:
            assert conn.our_state is CLOSED
            assert conn.their_state is CLOSED
        p = setup()
        with pytest.raises(RuntimeError):
            p.conn[who_shot_second].receive_data(b'123')
        p = setup()
        p.conn[who_shot_first].receive_data(b'GET')
        with pytest.raises(RemoteProtocolError):
            p.conn[who_shot_first].next_event()