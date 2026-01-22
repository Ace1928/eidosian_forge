from http import HTTPStatus
import pytest
from .. import _events
from .._events import (
from .._util import LocalProtocolError
def test_intenum_status_code() -> None:
    r = Response(status_code=HTTPStatus.OK, headers=[], http_version='1.0')
    assert r.status_code == HTTPStatus.OK
    assert type(r.status_code) is not type(HTTPStatus.OK)
    assert type(r.status_code) is int