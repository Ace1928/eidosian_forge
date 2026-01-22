from typing import Any, cast, Dict, List, Optional, Tuple, Type
import pytest
from .._connection import _body_framing, _keep_alive, Connection, NEED_DATA, PAUSED
from .._events import (
from .._state import (
from .._util import LocalProtocolError, RemoteProtocolError, Sentinel
from .helpers import ConnectionPair, get_all_events, receive_and_get
def test__body_framing() -> None:

    def headers(cl: Optional[int], te: bool) -> List[Tuple[str, str]]:
        headers = []
        if cl is not None:
            headers.append(('Content-Length', str(cl)))
        if te:
            headers.append(('Transfer-Encoding', 'chunked'))
        return headers

    def resp(status_code: int=200, cl: Optional[int]=None, te: bool=False) -> Response:
        return Response(status_code=status_code, headers=headers(cl, te))

    def req(cl: Optional[int]=None, te: bool=False) -> Request:
        h = headers(cl, te)
        h += [('Host', 'example.com')]
        return Request(method='GET', target='/', headers=h)
    for kwargs in [{}, {'cl': 100}, {'te': True}, {'cl': 100, 'te': True}]:
        kwargs = cast(Dict[str, Any], kwargs)
        for meth, r in [(b'HEAD', resp(**kwargs)), (b'GET', resp(status_code=204, **kwargs)), (b'GET', resp(status_code=304, **kwargs))]:
            assert _body_framing(meth, r) == ('content-length', (0,))
    for kwargs in [{'te': True}, {'cl': 100, 'te': True}]:
        kwargs = cast(Dict[str, Any], kwargs)
        for meth, r in [(None, req(**kwargs)), (b'GET', resp(**kwargs))]:
            assert _body_framing(meth, r) == ('chunked', ())
    for meth, r in [(None, req(cl=100)), (b'GET', resp(cl=100))]:
        assert _body_framing(meth, r) == ('content-length', (100,))
    assert _body_framing(None, req()) == ('content-length', (0,))
    assert _body_framing(b'GET', resp()) == ('http/1.0', ())