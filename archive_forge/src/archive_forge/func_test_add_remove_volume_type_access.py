import testtools
from openstack import exceptions
from openstack.tests.functional import base
def test_add_remove_volume_type_access(self):
    volume_type = self.operator_cloud.get_volume_type('test-volume-type')
    self.assertEqual('test-volume-type', volume_type.name)
    self.operator_cloud.add_volume_type_access('test-volume-type', self.operator_cloud.current_project_id)
    self._assert_project('test-volume-type', self.operator_cloud.current_project_id, allowed=True)
    self.operator_cloud.remove_volume_type_access('test-volume-type', self.operator_cloud.current_project_id)
    self._assert_project('test-volume-type', self.operator_cloud.current_project_id, allowed=False)