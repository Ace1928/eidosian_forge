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
def test_get_client_ssl_credentials_failure(self, mock_check_dca_metadata_path, mock_read_dca_metadata_file, mock_get_client_ssl_credentials, mock_ssl_channel_credentials):
    mock_check_dca_metadata_path.return_value = METADATA_PATH
    mock_read_dca_metadata_file.return_value = {'cert_provider_command': ['some command']}
    mock_get_client_ssl_credentials.side_effect = exceptions.ClientCertError()
    with pytest.raises(exceptions.MutualTLSChannelError):
        with mock.patch.dict(os.environ, {environment_vars.GOOGLE_API_USE_CLIENT_CERTIFICATE: 'true'}):
            assert google.auth.transport.grpc.SslCredentials().ssl_credentials