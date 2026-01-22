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
def test_reject_garbage_in_header_line() -> None:
    with pytest.raises(LocalProtocolError):
        tr(READERS[CLIENT, IDLE], b'HEAD /foo HTTP/1.1\r\nHost: foo\x00bar\r\n\r\n', None)