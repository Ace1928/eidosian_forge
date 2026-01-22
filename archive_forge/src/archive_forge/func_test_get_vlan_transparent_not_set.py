from neutron_lib.api.definitions import vlantransparent
from neutron_lib.tests.unit.api.definitions import base
def test_get_vlan_transparent_not_set(self):
    self.assertFalse(vlantransparent.get_vlan_transparent({'vlanxtx': True, 'vlan': '1'}))