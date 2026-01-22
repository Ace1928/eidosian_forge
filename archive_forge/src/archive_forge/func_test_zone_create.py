from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, 'zone_exists')
def test_zone_create(self, mock_zone_exists):
    mock_zone_exists.return_value = False
    zone_manager = self._dnsutils._dns_manager.MicrosoftDNS_Zone
    zone_manager.CreateZone.return_value = (mock.sentinel.zone_path,)
    zone_path = self._dnsutils.zone_create(zone_name=mock.sentinel.zone_name, zone_type=mock.sentinel.zone_type, ds_integrated=mock.sentinel.ds_integrated, data_file_name=mock.sentinel.data_file_name, ip_addrs=mock.sentinel.ip_addrs, admin_email_name=mock.sentinel.admin_email_name)
    zone_manager.CreateZone.assert_called_once_with(ZoneName=mock.sentinel.zone_name, ZoneType=mock.sentinel.zone_type, DsIntegrated=mock.sentinel.ds_integrated, DataFileName=mock.sentinel.data_file_name, IpAddr=mock.sentinel.ip_addrs, AdminEmailname=mock.sentinel.admin_email_name)
    mock_zone_exists.assert_called_once_with(mock.sentinel.zone_name)
    self.assertEqual(mock.sentinel.zone_path, zone_path)