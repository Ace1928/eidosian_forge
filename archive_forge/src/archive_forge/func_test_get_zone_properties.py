from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, '_get_zone')
def test_get_zone_properties(self, mock_get_zone):
    mock_get_zone.return_value = mock.Mock(ZoneType=mock.sentinel.zone_type, DsIntegrated=mock.sentinel.ds_integrated, DataFile=mock.sentinel.data_file_name, MasterServers=[mock.sentinel.ip_addrs])
    zone_properties = self._dnsutils.get_zone_properties(mock.sentinel.zone_name)
    expected_zone_props = {'zone_type': mock.sentinel.zone_type, 'ds_integrated': mock.sentinel.ds_integrated, 'master_servers': [mock.sentinel.ip_addrs], 'data_file_name': mock.sentinel.data_file_name}
    self.assertEqual(expected_zone_props, zone_properties)
    mock_get_zone.assert_called_once_with(mock.sentinel.zone_name, ignore_missing=False)