from neutron_lib.api.definitions import vlantransparent
from neutron_lib.tests.unit.api.definitions import base
def test_get_vlan_transparent(self):
    self.assertTrue(vlantransparent.get_vlan_transparent({vlantransparent.VLANTRANSPARENT: True, 'vlan': '1'}))