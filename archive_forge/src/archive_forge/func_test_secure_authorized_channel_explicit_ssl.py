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
def test_secure_authorized_channel_explicit_ssl(self, secure_channel, ssl_channel_credentials, metadata_call_credentials, composite_channel_credentials, get_client_ssl_credentials):
    credentials = mock.Mock()
    request = mock.Mock()
    target = 'example.com:80'
    ssl_credentials = mock.Mock()
    google.auth.transport.grpc.secure_authorized_channel(credentials, request, target, ssl_credentials=ssl_credentials)
    assert not get_client_ssl_credentials.called
    assert not ssl_channel_credentials.called
    composite_channel_credentials.assert_called_once_with(ssl_credentials, metadata_call_credentials.return_value)