import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_ConnectionState_double_protocol_switch() -> None:
    for server_switch in [None, _SWITCH_UPGRADE, _SWITCH_CONNECT]:
        cs = ConnectionState()
        cs.process_client_switch_proposal(_SWITCH_UPGRADE)
        cs.process_client_switch_proposal(_SWITCH_CONNECT)
        cs.process_event(CLIENT, Request)
        cs.process_event(CLIENT, EndOfMessage)
        assert cs.states == {CLIENT: MIGHT_SWITCH_PROTOCOL, SERVER: SEND_RESPONSE}
        cs.process_event(SERVER, _response_type_for_switch[server_switch], server_switch)
        if server_switch is None:
            assert cs.states == {CLIENT: DONE, SERVER: SEND_BODY}
        else:
            assert cs.states == {CLIENT: SWITCHED_PROTOCOL, SERVER: SWITCHED_PROTOCOL}