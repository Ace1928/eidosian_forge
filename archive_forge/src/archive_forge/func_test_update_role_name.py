import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_update_role_name(self):
    role = fixtures.Role(self.client, domain=self.project_domain_id)
    self.useFixture(role)
    new_name = fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex
    role_ret = self.client.roles.update(role.id, name=new_name)
    role.ref.update({'name': new_name})
    self.check_role(role_ret, role.ref)