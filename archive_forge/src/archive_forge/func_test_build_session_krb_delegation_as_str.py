import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
def test_build_session_krb_delegation_as_str(self):
    winrm_transport = transport.Transport(endpoint='Endpoint', server_cert_validation='validate', username='test', password='test', auth_method='kerberos', kerberos_delegation='True')
    self.assertTrue(winrm_transport.kerberos_delegation)