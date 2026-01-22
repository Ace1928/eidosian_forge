import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_revoke_role_group_id_project_not_exists(self):
    uris = self._get_mock_role_query_urls(self.role_data, project_data=self.project_data, group_data=self.group_data)
    uris.extend([dict(method='HEAD', uri=self.get_mock_url(resource='projects', append=[self.project_data.project_id, 'groups', self.group_data.group_id, 'roles', self.role_data.role_id]), complete_qs=True, status_code=404)])
    self.register_uris(uris)
    self.assertFalse(self.cloud.revoke_role(self.role_data.role_id, group=self.group_data.group_id, project=self.project_data.project_id))
    self.assert_calls()