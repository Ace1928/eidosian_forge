from typing import Any, Callable, Generator, List
import pytest
from .._events import (
from .._headers import Headers, normalize_and_validate
from .._readers import (
from .._receivebuffer import ReceiveBuffer
from .._state import (
from .._util import LocalProtocolError
from .._writers import (
from .helpers import normalize_data_events
def test_reject_garbage_after_request_line() -> None:
    with pytest.raises(LocalProtocolError):
        tr(READERS[SERVER, SEND_RESPONSE], b'HTTP/1.0 200 OK\x00xxxx\r\n\r\n', None)