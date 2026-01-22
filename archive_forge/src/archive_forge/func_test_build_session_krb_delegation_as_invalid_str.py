import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_krb_delegation_as_invalid_str(self):
    with self.assertRaises(ValueError) as exc:
        transport.Transport(endpoint='Endpoint', server_cert_validation='validate', username='test', password='test', auth_method='kerberos', kerberos_delegation='invalid_value')
    self.assertEqual("invalid truth value 'invalid_value'", str(exc.exception))