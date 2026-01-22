import datetime
import os
import time
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import credentials
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('google.auth.transport._mtls_helper._read_dca_metadata_file', autospec=True)
@mock.patch('google.auth.transport._mtls_helper._check_dca_metadata_path', autospec=True)
def test_secure_authorized_channel_with_client_cert_callback_failure(self, check_dca_metadata_path, read_dca_metadata_file, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
    credentials = mock.Mock()
    request = mock.Mock()
    target = 'example.com:80'
    client_cert_callback = mock.Mock()
    client_cert_callback.side_effect = Exception('callback exception')
    with pytest.raises(Exception) as excinfo:
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, client_cert_callback=client_cert_callback)
    assert str(excinfo.value) == 'callback exception'