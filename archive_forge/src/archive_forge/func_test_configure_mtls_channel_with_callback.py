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
def test_configure_mtls_channel_with_callback(self, mock_make_mutual_tls_http):
    callback = mock.Mock()
    callback.return_value = (pytest.public_cert_bytes, pytest.private_key_bytes)
    authed_http = google.auth.transport.urllib3.AuthorizedHttp(credentials=mock.Mock(), http=mock.Mock())
    with pytest.warns(UserWarning):
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            is_mtls = authed_http.configure_mtls_channel(callback)
    assert is_mtls
    mock_make_mutual_tls_http.assert_called_once_with(cert=pytest.public_cert_bytes, key=pytest.private_key_bytes)