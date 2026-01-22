import pprint
from openstack import exceptions
from openstack.tests.functional import base
def test_delete_project_not_found(self):
    self.assertFalse(self.operator_cloud.delete_project('doesNotExist'))