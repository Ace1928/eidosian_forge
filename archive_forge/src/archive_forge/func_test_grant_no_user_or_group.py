import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_grant_no_user_or_group(self):
    uris = self.__get('role', self.role_data, 'role_name', [], use_name=True)
    uris.extend(self.__user_mocks(self.user_data, use_name=True, is_found=False))
    self.register_uris(uris)
    with testtools.ExpectedException(exceptions.SDKException, 'Must specify either a user or a group'):
        self.cloud.grant_role(self.role_data.role_name, user=self.user_data.name)
    self.assert_calls()