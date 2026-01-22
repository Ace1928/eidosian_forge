import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_ConnectionState_protocol_switch_accepted() -> None:
    for switch_event in [_SWITCH_UPGRADE, _SWITCH_CONNECT]:
        cs = ConnectionState()
        cs.process_client_switch_proposal(switch_event)
        cs.process_event(CLIENT, Request)
        cs.process_event(CLIENT, Data)
        assert cs.states == {CLIENT: SEND_BODY, SERVER: SEND_RESPONSE}
        cs.process_event(CLIENT, EndOfMessage)
        assert cs.states == {CLIENT: MIGHT_SWITCH_PROTOCOL, SERVER: SEND_RESPONSE}
        cs.process_event(SERVER, InformationalResponse)
        assert cs.states == {CLIENT: MIGHT_SWITCH_PROTOCOL, SERVER: SEND_RESPONSE}
        cs.process_event(SERVER, _response_type_for_switch[switch_event], switch_event)
        assert cs.states == {CLIENT: SWITCHED_PROTOCOL, SERVER: SWITCHED_PROTOCOL}