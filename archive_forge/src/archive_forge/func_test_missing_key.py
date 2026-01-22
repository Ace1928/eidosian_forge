import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
@mock.patch('subprocess.Popen', autospec=True)
def test_missing_key(self, mock_popen):
    mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes, b'')
    with pytest.raises(exceptions.ClientCertError):
        _mtls_helper._run_cert_provider_command(['command'])
    mock_popen.return_value = self.create_mock_process(pytest.public_cert_bytes + PASSPHRASE, b'')
    with pytest.raises(exceptions.ClientCertError):
        _mtls_helper._run_cert_provider_command(['command'], expect_encrypted_key=True)