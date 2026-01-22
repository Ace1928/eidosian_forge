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
def t_body_reader(thunk: Any, data: bytes, expected: Any, do_eof: bool=False) -> None:
    print('Test 1')
    buf = makebuf(data)
    assert _run_reader(thunk(), buf, do_eof) == expected
    print('Test 2')
    reader = thunk()
    buf = ReceiveBuffer()
    events = []
    for i in range(len(data)):
        events += _run_reader(reader, buf, False)
        buf += data[i:i + 1]
    events += _run_reader(reader, buf, do_eof)
    assert normalize_data_events(events) == expected
    is_complete = any((type(event) is EndOfMessage for event in expected))
    if is_complete and (not do_eof):
        buf = makebuf(data + b'trailing')
        assert _run_reader(thunk(), buf, False) == expected