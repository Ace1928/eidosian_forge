import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_segment_create_delete(self):
    name = uuid.uuid4().hex
    json_output = self.openstack(' network segment create ' + '--network ' + self.NETWORK_ID + ' ' + '--network-type geneve ' + '--segment 2055 ' + name, parse_output=True)
    self.assertEqual(name, json_output['name'])
    raw_output = self.openstack('network segment delete ' + name)
    self.assertOutput('', raw_output)