import uuid
from openstackclient.tests.functional.network.v2 import common
def test_address_group_create_and_delete(self):
    """Test create, delete multiple"""
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('address group create ' + name1, parse_output=True)
    self.assertEqual(name1, cmd_output['name'])
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('address group create ' + name2, parse_output=True)
    self.assertEqual(name2, cmd_output['name'])
    raw_output = self.openstack('address group delete ' + name1 + ' ' + name2)
    self.assertOutput('', raw_output)