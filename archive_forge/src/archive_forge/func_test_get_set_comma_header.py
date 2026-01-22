import pytest
from .._events import Request
from .._headers import (
from .._util import LocalProtocolError
def test_get_set_comma_header() -> None:
    headers = normalize_and_validate([('Connection', 'close'), ('whatever', 'something'), ('connectiON', 'fOo,, , BAR')])
    assert get_comma_header(headers, b'connection') == [b'close', b'foo', b'bar']
    headers = set_comma_header(headers, b'newthing', ['a', 'b'])
    with pytest.raises(LocalProtocolError):
        set_comma_header(headers, b'newthing', ['  a', 'b'])
    assert headers == [(b'connection', b'close'), (b'whatever', b'something'), (b'connection', b'fOo,, , BAR'), (b'newthing', b'a'), (b'newthing', b'b')]
    headers = set_comma_header(headers, b'whatever', ['different thing'])
    assert headers == [(b'connection', b'close'), (b'connection', b'fOo,, , BAR'), (b'newthing', b'a'), (b'newthing', b'b'), (b'whatever', b'different thing')]