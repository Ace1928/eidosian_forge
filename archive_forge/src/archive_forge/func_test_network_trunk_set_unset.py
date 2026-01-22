import json
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_trunk_set_unset(self):
    trunk_name = uuid.uuid4().hex
    json_output = json.loads(self.openstack('network trunk create %s --parent-port %s -f json ' % (trunk_name, self.parent_port_name)))
    self.addCleanup(self.openstack, 'network trunk delete ' + trunk_name)
    self.assertEqual(trunk_name, json_output['name'])
    self.openstack('network trunk set --enable ' + trunk_name)
    json_output = json.loads(self.openstack('network trunk show -f json ' + trunk_name))
    self.assertTrue(json_output['is_admin_state_up'])
    self.openstack('network trunk set ' + '--subport port=%s,segmentation-type=vlan,segmentation-id=42 ' % self.sub_port_name + trunk_name)
    json_output = json.loads(self.openstack('network trunk show -f json ' + trunk_name))
    self.assertEqual([{'port_id': self.sub_port_id, 'segmentation_id': 42, 'segmentation_type': 'vlan'}], json_output['sub_ports'])
    self.openstack('network trunk unset ' + trunk_name + ' --subport ' + self.sub_port_name)
    json_output = json.loads(self.openstack('network trunk show -f json ' + trunk_name))
    self.assertEqual([], json_output['sub_ports'])