import datetime
import functools
import os
import sys
import freezegun
import mock
import OpenSSL
import pytest  # type: ignore
import requests
import requests.adapters
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._custom_tls_signer
import google.auth.transport._mtls_helper
import google.auth.transport.requests
from google.oauth2 import service_account
from tests.transport import compliance
def test_request_default_timeout(self):
    credentials = mock.Mock(wraps=CredentialsStub())
    response = make_response()
    adapter = AdapterStub([response])
    authed_session = google.auth.transport.requests.AuthorizedSession(credentials)
    authed_session.mount(self.TEST_URL, adapter)
    patcher = mock.patch('google.auth.transport.requests.requests.Session.request')
    with patcher as patched_request:
        authed_session.request('GET', self.TEST_URL)
    expected_timeout = google.auth.transport.requests._DEFAULT_TIMEOUT
    assert patched_request.call_args[1]['timeout'] == expected_timeout