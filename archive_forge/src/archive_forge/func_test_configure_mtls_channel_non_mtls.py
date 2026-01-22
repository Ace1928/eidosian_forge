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
@mock.patch('google.auth.transport.urllib3._make_mutual_tls_http', autospec=True)
@mock.patch('google.auth.transport._mtls_helper.get_client_cert_and_key', autospec=True)
def test_configure_mtls_channel_non_mtls(self, mock_get_client_cert_and_key, mock_make_mutual_tls_http):
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials=mock.Mock())
    mock_get_client_cert_and_key.return_value = (False, None, None)
    with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
        is_mtls = authed_http.configure_mtls_channel()
    assert not is_mtls
    mock_get_client_cert_and_key.assert_called_once()
    mock_make_mutual_tls_http.assert_not_called()