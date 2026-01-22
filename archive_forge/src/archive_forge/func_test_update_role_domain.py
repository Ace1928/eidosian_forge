import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_update_role_domain(self):
    role = fixtures.Role(self.client)
    self.useFixture(role)
    domain = fixtures.Domain(self.client)
    self.useFixture(domain)
    new_domain = domain.id
    role_ret = self.client.roles.update(role.id, domain=new_domain)
    role.ref.update({'domain': new_domain})
    self.check_role(role_ret, role.ref)