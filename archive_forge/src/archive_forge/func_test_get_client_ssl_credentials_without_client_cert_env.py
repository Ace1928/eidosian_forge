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
def test_get_client_ssl_credentials_without_client_cert_env(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file, mock_get_client_ssl_credentials, mock_ssl_channel_credentials):
    ssl_credentials = google.auth.transport.grpc.SslCredentials()
    assert ssl_credentials.ssl_credentials is not None
    assert not ssl_credentials.is_mtls
    mock_check_dca_metadata_path.assert_not_called()
    mock_read_dca_metadata_file.assert_not_called()
    mock_get_client_ssl_credentials.assert_not_called()
    mock_ssl_channel_credentials.assert_called_once()