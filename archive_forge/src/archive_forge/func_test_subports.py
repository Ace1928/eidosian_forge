from openstack.network.v2 import network
from openstack.network.v2 import port
from openstack.network.v2 import trunk as _trunk
from openstack.tests.functional import base
def test_subports(self):
    port_for_subport = self.user_cloud.network.create_port(network_id=self.NET_ID)
    self.ports_to_clean.append(port_for_subport.id)
    subports = [{'port_id': port_for_subport.id, 'segmentation_type': 'vlan', 'segmentation_id': 111}]
    sot = self.user_cloud.network.get_trunk_subports(self.TRUNK_ID)
    self.assertEqual({'sub_ports': []}, sot)
    self.user_cloud.network.add_trunk_subports(self.TRUNK_ID, subports)
    sot = self.user_cloud.network.get_trunk_subports(self.TRUNK_ID)
    self.assertEqual({'sub_ports': subports}, sot)
    self.user_cloud.network.delete_trunk_subports(self.TRUNK_ID, [{'port_id': port_for_subport.id}])
    sot = self.user_cloud.network.get_trunk_subports(self.TRUNK_ID)
    self.assertEqual({'sub_ports': []}, sot)