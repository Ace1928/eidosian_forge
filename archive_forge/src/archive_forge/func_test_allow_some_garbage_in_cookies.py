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
def test_allow_some_garbage_in_cookies() -> None:
    tr(READERS[CLIENT, IDLE], b'HEAD /foo HTTP/1.1\r\nHost: foo\r\nSet-Cookie: ___utmvafIumyLc=kUd\x01UpAt; path=/; Max-Age=900\r\n\r\n', Request(method='HEAD', target='/foo', headers=[('Host', 'foo'), ('Set-Cookie', '___utmvafIumyLc=kUd\x01UpAt; path=/; Max-Age=900')]))