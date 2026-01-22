import uuid
from openstackclient.tests.functional.volume.v1 import common
def test_volume_create_and_delete(self):
    """Test create, delete multiple"""
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 1 ' + name1, parse_output=True)
    self.assertEqual(1, cmd_output['size'])
    name2 = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 2 ' + name2, parse_output=True)
    self.assertEqual(2, cmd_output['size'])
    self.wait_for_status('volume', name1, 'available')
    self.wait_for_status('volume', name2, 'available')
    del_output = self.openstack('volume delete ' + name1 + ' ' + name2)
    self.assertOutput('', del_output)