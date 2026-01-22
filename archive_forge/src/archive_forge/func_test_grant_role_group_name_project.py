import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_grant_role_group_name_project(self):
    uris = self._get_mock_role_query_urls(self.role_data, project_data=self.project_data, group_data=self.group_data, use_role_name=True, use_group_name=True)
    uris.extend([dict(method='HEAD', uri=self.get_mock_url(resource='projects', append=[self.project_data.project_id, 'groups', self.group_data.group_id, 'roles', self.role_data.role_id]), complete_qs=True, status_code=404), dict(method='PUT', uri=self.get_mock_url(resource='projects', append=[self.project_data.project_id, 'groups', self.group_data.group_id, 'roles', self.role_data.role_id]), status_code=200)])
    self.register_uris(uris)
    self.assertTrue(self.cloud.grant_role(self.role_data.role_name, group=self.group_data.group_name, project=self.project_data.project_id))
    self.assert_calls()