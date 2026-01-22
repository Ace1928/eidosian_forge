from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, '_get_wmi_obj')
def test_dns_manager_fail(self, mock_get_wmi_obj):
    self._dnsutils._dns_manager_attr = None
    expected_exception = exceptions.DNSException
    mock_get_wmi_obj.side_effect = expected_exception
    self.assertRaises(expected_exception, lambda: self._dnsutils._dns_manager)
    mock_get_wmi_obj.assert_called_once_with(self._dnsutils._DNS_NAMESPACE % self._dnsutils._host)