from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
@mock.patch.object(nvgreutils.NvgreUtils, '_get_network_iface_index')
def test_create_provider_route(self, mock_get_iface_index):
    mock_get_iface_index.return_value = mock.sentinel.iface_index
    self.utils._scimv2.MSFT_NetVirtualizationProviderRouteSettingData = mock.MagicMock(return_value=[])
    self.utils.create_provider_route(mock.sentinel.fake_network)
    scimv2 = self.utils._scimv2
    obj_class = scimv2.MSFT_NetVirtualizationProviderRouteSettingData
    obj_class.new.assert_called_once_with(InterfaceIndex=mock.sentinel.iface_index, DestinationPrefix='%s/0' % constants.IPV4_DEFAULT, NextHop=constants.IPV4_DEFAULT)