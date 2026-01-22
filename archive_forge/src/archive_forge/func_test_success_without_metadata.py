import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
@mock.patch('google.auth.transport._mtls_helper._check_dca_metadata_path', autospec=True)
def test_success_without_metadata(self, mock_check_dca_metadata_path):
    mock_check_dca_metadata_path.return_value = False
    has_cert, cert, key, passphrase = _mtls_helper.get_client_ssl_credentials()
    assert not has_cert
    assert cert is None
    assert key is None
    assert passphrase is None