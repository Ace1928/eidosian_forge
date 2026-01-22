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
def test_Http10Writer() -> None:
    w = Http10Writer()
    assert dowrite(w, Data(data=b'1234')) == b'1234'
    assert dowrite(w, EndOfMessage()) == b''
    with pytest.raises(LocalProtocolError):
        dowrite(w, EndOfMessage(headers=[('Etag', 'asdf')]))