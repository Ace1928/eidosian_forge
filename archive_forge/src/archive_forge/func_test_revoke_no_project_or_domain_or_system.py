import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_revoke_no_project_or_domain_or_system(self):
    uris = self._get_mock_role_query_urls(self.role_data, user_data=self.user_data, use_role_name=True, use_user_name=True)
    self.register_uris(uris)
    with testtools.ExpectedException(exceptions.SDKException, 'Must specify either a domain, project or system'):
        self.cloud.revoke_role(self.role_data.role_name, user=self.user_data.name)
    self.assert_calls()