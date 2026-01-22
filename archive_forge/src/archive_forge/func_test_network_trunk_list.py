import json
import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_trunk_list(self):
    trunk_name = uuid.uuid4().hex
    json_output = json.loads(self.openstack('network trunk create %s --parent-port %s -f json ' % (trunk_name, self.parent_port_name)))
    self.addCleanup(self.openstack, 'network trunk delete ' + trunk_name)
    self.assertEqual(trunk_name, json_output['name'])
    json_output = json.loads(self.openstack('network trunk list -f json'))
    self.assertIn(trunk_name, [tr['Name'] for tr in json_output])