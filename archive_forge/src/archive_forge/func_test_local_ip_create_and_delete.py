import uuid
from openstackclient.tests.functional.network.v2 import common
def test_local_ip_create_and_delete(self):
    """Test create, delete multiple"""
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('local ip create ' + name1, parse_output=True)
    self.assertEqual(name1, cmd_output['name'])
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('local ip create ' + name2, parse_output=True)
    self.assertEqual(name2, cmd_output['name'])
    raw_output = self.openstack('local ip delete ' + name1 + ' ' + name2)
    self.assertOutput('', raw_output)