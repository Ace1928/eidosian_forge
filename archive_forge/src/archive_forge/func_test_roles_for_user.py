import uuid
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import roles
def test_roles_for_user(self):
    self.stub_url('GET', ['users', 'foo', 'roles'], json=self.TEST_ROLES)
    role_list = self.client.roles.roles_for_user('foo')
    [self.assertIsInstance(r, roles.Role) for r in role_list]