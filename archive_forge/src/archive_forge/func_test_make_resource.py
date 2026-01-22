from unittest import mock
from keystoneauth1 import adapter
from openstack.block_storage.v3 import group_type
from openstack.tests.unit import base
def test_make_resource(self):
    resource = group_type.GroupType(**GROUP_TYPE)
    self.assertEqual(GROUP_TYPE['id'], resource.id)
    self.assertEqual(GROUP_TYPE['name'], resource.name)
    self.assertEqual(GROUP_TYPE['description'], resource.description)
    self.assertEqual(GROUP_TYPE['is_public'], resource.is_public)
    self.assertEqual(GROUP_TYPE['group_specs'], resource.group_specs)