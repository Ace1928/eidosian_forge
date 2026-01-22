import json
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_trunk_list_subports(self):
    trunk_name = uuid.uuid4().hex
    json_output = json.loads(self.openstack('network trunk create %s --parent-port %s --subport port=%s,segmentation-type=vlan,segmentation-id=42 -f json ' % (trunk_name, self.parent_port_name, self.sub_port_name)))
    self.addCleanup(self.openstack, 'network trunk delete ' + trunk_name)
    self.assertEqual(trunk_name, json_output['name'])
    json_output = json.loads(self.openstack('network subport list --trunk %s -f json' % trunk_name))
    self.assertEqual([{'Port': self.sub_port_id, 'Segmentation ID': 42, 'Segmentation Type': 'vlan'}], json_output)