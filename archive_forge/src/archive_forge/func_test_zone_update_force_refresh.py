from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.dns import dnsutils
@mock.patch.object(dnsutils.DNSUtils, '_get_zone')
def test_zone_update_force_refresh(self, mock_get_zone):
    mock_zone = mock.MagicMock(DsIntegrated=False, ZoneType=constants.DNS_ZONE_TYPE_SECONDARY)
    mock_get_zone.return_value = mock_zone
    self._dnsutils.zone_update(mock.sentinel.zone_name)
    mock_get_zone.assert_called_once_with(mock.sentinel.zone_name, ignore_missing=False)
    mock_zone.ForceRefresh.assert_called_once_with()