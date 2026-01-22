import aiohttp  # type: ignore
from aioresponses import aioresponses, core  # type: ignore
import mock
import pytest  # type: ignore
from tests_async.transport import async_compliance
import google.auth._credentials_async
from google.auth.transport import _aiohttp_requests as aiohttp_requests
import google.auth.transport._mtls_helper
def test_constructor_with_auth_request(self):
    http = mock.create_autospec(aiohttp.ClientSession, instance=True, _auto_decompress=False)
    auth_request = aiohttp_requests.Request(http)
    authed_session = aiohttp_requests.AuthorizedSession(mock.sentinel.credentials, auth_request=auth_request)
    assert authed_session._auth_request == auth_request