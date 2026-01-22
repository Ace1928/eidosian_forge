import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_revoke_role_group_name_domain(self):
    uris = self._get_mock_role_query_urls(self.role_data, domain_data=self.domain_data, group_data=self.group_data, use_role_name=True, use_group_name=True)
    uris.extend([dict(method='HEAD', uri=self.get_mock_url(resource='domains', append=[self.domain_data.domain_id, 'groups', self.group_data.group_id, 'roles', self.role_data.role_id]), complete_qs=True, status_code=204), dict(method='DELETE', uri=self.get_mock_url(resource='domains', append=[self.domain_data.domain_id, 'groups', self.group_data.group_id, 'roles', self.role_data.role_id]), status_code=200)])
    self.register_uris(uris)
    self.assertTrue(self.cloud.revoke_role(self.role_data.role_name, group=self.group_data.group_name, domain=self.domain_data.domain_id))
    self.assert_calls()