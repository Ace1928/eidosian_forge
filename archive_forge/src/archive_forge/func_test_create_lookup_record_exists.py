from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.network import nvgreutils
def test_create_lookup_record_exists(self):
    lookup = mock.MagicMock(VirtualSubnetID=mock.sentinel.fake_vsid, ProviderAddress=mock.sentinel.provider_addr, CustomerAddress=mock.sentinel.customer_addr, MACAddress=mock.sentinel.mac_addr)
    scimv2 = self.utils._scimv2
    obj_class = scimv2.MSFT_NetVirtualizationLookupRecordSettingData
    obj_class.return_value = [lookup]
    self.utils.create_lookup_record(mock.sentinel.provider_addr, mock.sentinel.customer_addr, mock.sentinel.mac_addr, mock.sentinel.fake_vsid)
    self.assertFalse(obj_class.new.called)