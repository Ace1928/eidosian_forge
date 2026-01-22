import uuid
import fixtures
from openstackclient.tests.functional.image import base
def test_image_attributes(self):
    """Test set, unset, show on attributes, tags and properties"""
    self.openstack('image set ' + '--min-disk 4 ' + '--min-ram 5 ' + '--public ' + self.name)
    output = self.openstack('image show ' + self.name, parse_output=True)
    self.assertEqual(4, output['min_disk'])
    self.assertEqual(5, output['min_ram'])
    self.assertEqual('public', output['visibility'])
    self.openstack('image set ' + '--property a=b ' + '--property c=d ' + '--property hw_rng_model=virtio ' + '--public ' + self.name)
    output = self.openstack('image show ' + self.name, parse_output=True)
    self.assertIn('a', output['properties'])
    self.assertIn('c', output['properties'])
    self.openstack('image unset ' + '--property a ' + '--property c ' + '--property hw_rng_model ' + self.name)
    output = self.openstack('image show ' + self.name, parse_output=True)
    self.assertNotIn('a', output['properties'])
    self.assertNotIn('c', output['properties'])
    self.assertNotIn('01', output['tags'])
    self.openstack('image set ' + '--tag 01 ' + self.name)
    output = self.openstack('image show ' + self.name, parse_output=True)
    self.assertIn('01', output['tags'])
    self.openstack('image unset ' + '--tag 01 ' + self.name)
    output = self.openstack('image show ' + self.name, parse_output=True)
    self.assertNotIn('01', output['tags'])