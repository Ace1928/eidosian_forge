from openstack.network.v2 import address_group as _address_group
from openstack.tests.functional import base
def test_add_remove_addresses(self):
    addrs = ['127.0.0.1/32', 'fe80::/10']
    sot = self.user_cloud.network.add_addresses_to_address_group(self.ADDRESS_GROUP_ID, addrs)
    updated_addrs = self.ADDRESSES.copy()
    updated_addrs.extend(addrs)
    self.assertCountEqual(updated_addrs, sot.addresses)
    sot = self.user_cloud.network.remove_addresses_from_address_group(self.ADDRESS_GROUP_ID, addrs)
    self.assertCountEqual(self.ADDRESSES, sot.addresses)