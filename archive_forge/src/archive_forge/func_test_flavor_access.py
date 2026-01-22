import uuid
from openstack import exceptions
from openstack.tests.functional import base
def test_flavor_access(self):
    flavor_name = uuid.uuid4().hex
    flv = self.operator_cloud.compute.create_flavor(is_public=False, name=flavor_name, ram=128, vcpus=1, disk=0)
    self.addCleanup(self.conn.compute.delete_flavor, flv.id)
    flv_cmp = self.user_cloud.compute.find_flavor(flavor_name)
    self.assertIsNone(flv_cmp)
    flv_cmp = self.operator_cloud.compute.find_flavor(flavor_name)
    self.assertIsNotNone(flv_cmp)
    self.assertEqual(flavor_name, flv_cmp.name)
    project = self.operator_cloud.get_project('demo')
    self.assertIsNotNone(project)
    self.operator_cloud.compute.flavor_add_tenant_access(flv.id, project['id'])
    flv_cmp = self.user_cloud.compute.find_flavor(flavor_name)
    self.assertIsNotNone(flv_cmp)
    self.operator_cloud.compute.flavor_remove_tenant_access(flv.id, project['id'])
    flv_cmp = self.user_cloud.compute.find_flavor(flavor_name)
    self.assertIsNone(flv_cmp)