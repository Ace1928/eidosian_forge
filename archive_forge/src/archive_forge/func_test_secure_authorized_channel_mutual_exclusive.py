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
def test_secure_authorized_channel_mutual_exclusive(self, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
    credentials = mock.Mock()
    request = mock.Mock()
    target = 'example.com:80'
    ssl_credentials = mock.Mock()
    client_cert_callback = mock.Mock()
    with pytest.raises(ValueError):
        google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, ssl_credentials=ssl_credentials, client_cert_callback=client_cert_callback)