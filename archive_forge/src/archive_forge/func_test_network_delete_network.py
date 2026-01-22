import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_delete_network(self):
    """Test create, delete multiple"""
    if not self.haz_network:
        self.skipTest('No Network service present')
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('network create ' + '--description aaaa ' + name1, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual('aaaa', cmd_output['description'])
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('network create ' + '--description bbbb ' + name2, parse_output=True)
    self.assertIsNotNone(cmd_output['id'])
    self.assertEqual('bbbb', cmd_output['description'])
    del_output = self.openstack('network delete %s %s' % (name1, name2))
    self.assertOutput('', del_output)