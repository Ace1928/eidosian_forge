import uuid
from openstackclient.tests.functional.volume.v1 import common
def test_volume_create_and_list_and_show_backward_compatibility(self):
    """Test backward compatibility of create, list, show"""
    name1 = uuid.uuid4().hex
    output = self.openstack('volume create ' + '-c display_name -c id ' + '--size 1 ' + name1, parse_output=True)
    self.assertIn('display_name', output)
    self.assertEqual(name1, output['display_name'])
    self.assertIn('id', output)
    volume_id = output['id']
    self.assertIsNotNone(volume_id)
    self.assertNotIn('name', output)
    self.addCleanup(self.openstack, 'volume delete ' + volume_id)
    self.wait_for_status('volume', name1, 'available')
    output = self.openstack('volume list ' + '-c "Display Name"', parse_output=True)
    for each_volume in output:
        self.assertIn('Display Name', each_volume)
    output = self.openstack('volume list ' + '-c "Name"', parse_output=True)
    for each_volume in output:
        self.assertIn('Name', each_volume)
    output = self.openstack('volume show ' + '-c display_name -c id ' + name1, parse_output=True)
    self.assertIn('display_name', output)
    self.assertEqual(name1, output['display_name'])
    self.assertIn('id', output)
    self.assertNotIn('name', output)