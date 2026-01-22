import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
@mock.patch('subprocess.Popen', autospec=True)
def test_cert_provider_returns_error(self, mock_popen):
    mock_popen.return_value = self.create_mock_process(b'', b'some error')
    mock_popen.return_value.returncode = 1
    with pytest.raises(exceptions.ClientCertError):
        _mtls_helper._run_cert_provider_command(['command'])