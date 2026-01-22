import os
import mock
import unittest
from winrm import transport
from winrm.exceptions import WinRMError, InvalidCredentialsError
@mock.patch('requests.Session')
def test_close_session(self, mock_session):
    t_default = transport.Transport(endpoint='Endpoint', server_cert_validation='ignore', username='test', password='test', auth_method='basic')
    t_default.build_session()
    t_default.close_session()
    mock_session.return_value.close.assert_called_once_with()
    self.assertIsNone(t_default.session)