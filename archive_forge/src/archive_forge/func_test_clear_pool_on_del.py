import os
import sys
import mock
import OpenSSL
import pytest  # type: ignore
from six.moves import http_client
import urllib3  # type: ignore
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._mtls_helper
import google.auth.transport.urllib3
from google.oauth2 import service_account
from tests.transport import compliance
def test_clear_pool_on_del(self):
    http = mock.create_autospec(urllib3.PoolManager)
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(mock.sentinel.credentials, http=http)
    authed_http.__del__()
    http.clear.assert_called_with()
    authed_http.http = None
    authed_http.__del__()