from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
def test_zone_list(self):
    zone_manager = self._dnsutils._dns_manager.MicrosoftDNS_Zone
    zone_manager.return_value = [mock.Mock(Name=mock.sentinel.fake_name1), mock.Mock(Name=mock.sentinel.fake_name2)]
    zone_list = self._dnsutils.zone_list()
    expected_zone_list = [mock.sentinel.fake_name1, mock.sentinel.fake_name2]
    self.assertEqual(expected_zone_list, zone_list)
    zone_manager.assert_called_once_with()