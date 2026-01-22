import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_grant_role_user_id_domain(self):
    uris = self._get_mock_role_query_urls(self.role_data, domain_data=self.domain_data, user_data=self.user_data, use_role_name=True)
    uris.extend([dict(method='HEAD', uri=self.get_mock_url(resource='domains', append=[self.domain_data.domain_id, 'users', self.user_data.user_id, 'roles', self.role_data.role_id]), complete_qs=True, status_code=404), dict(method='PUT', uri=self.get_mock_url(resource='domains', append=[self.domain_data.domain_id, 'users', self.user_data.user_id, 'roles', self.role_data.role_id]), status_code=200)])
    self.register_uris(uris)
    self.assertTrue(self.cloud.grant_role(self.role_data.role_name, user=self.user_data.user_id, domain=self.domain_data.domain_id))
    self.assert_calls()