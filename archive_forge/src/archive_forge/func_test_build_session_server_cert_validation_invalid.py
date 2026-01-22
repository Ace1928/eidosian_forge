import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_server_cert_validation_invalid(self):
    with self.assertRaises(WinRMError) as exc:
        transport.Transport(endpoint='Endpoint', server_cert_validation='invalid_value', username='test', password='test', auth_method='basic')
    self.assertEqual('invalid server_cert_validation mode: invalid_value', str(exc.exception))