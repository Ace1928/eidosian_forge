import aiohttp  # type: ignore
from aioresponses import aioresponses, core  # type: ignore
import mock
import pytest  # type: ignore
from tests_async.transport import async_compliance
import google.auth._credentials_async
from google.auth.transport import _aiohttp_requests as aiohttp_requests
import google.auth.transport._mtls_helper
def make_with_parameter_request(self):
    http = aiohttp.ClientSession(auto_decompress=False)
    return aiohttp_requests.Request(http)