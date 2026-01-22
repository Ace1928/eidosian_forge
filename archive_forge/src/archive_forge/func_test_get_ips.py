from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
@mock.patch.object(_utils.socket, 'getaddrinfo')
def test_get_ips(self, mock_getaddrinfo):
    ips = ['1.2.3.4', '5.6.7.8']
    mock_getaddrinfo.return_value = [(None, None, None, None, (ip, 0)) for ip in ips]
    resulted_ips = _utils.get_ips(mock.sentinel.addr)
    self.assertEqual(ips, resulted_ips)
    mock_getaddrinfo.assert_called_once_with(mock.sentinel.addr, None, 0, 0, 0)