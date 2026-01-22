from openstack.block_storage.v2 import extension
from openstack.tests.unit import base
def test_make_extension(self):
    extension_resource = extension.Extension(**EXTENSION)
    self.assertEqual(EXTENSION['alias'], extension_resource.alias)
    self.assertEqual(EXTENSION['description'], extension_resource.description)
    self.assertEqual(EXTENSION['links'], extension_resource.links)
    self.assertEqual(EXTENSION['name'], extension_resource.name)
    self.assertEqual(EXTENSION['namespace'], extension_resource.namespace)
    self.assertEqual(EXTENSION['updated'], extension_resource.updated_at)