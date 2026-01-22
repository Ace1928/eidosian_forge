import pytest
from .._events import (
from .._state import (
from .._util import LocalProtocolError
def test_server_request_is_illegal() -> None:
    cs = ConnectionState()
    with pytest.raises(LocalProtocolError):
        cs.process_event(SERVER, Request)