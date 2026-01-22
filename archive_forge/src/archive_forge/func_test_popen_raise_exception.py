import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
@mock.patch('subprocess.Popen', autospec=True)
def test_popen_raise_exception(self, mock_popen):
    mock_popen.side_effect = OSError()
    with pytest.raises(exceptions.ClientCertError):
        _mtls_helper._run_cert_provider_command(['command'])