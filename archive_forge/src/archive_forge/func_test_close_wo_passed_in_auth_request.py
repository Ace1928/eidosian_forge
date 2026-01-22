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
def test_close_wo_passed_in_auth_request(self):
    authed_session = google.auth.transport.requests.AuthorizedSession(mock.sentinel.credentials)
    authed_session._auth_request_session = mock.Mock(spec=['close'])
    authed_session.close()
    authed_session._auth_request_session.close.assert_called_once_with()