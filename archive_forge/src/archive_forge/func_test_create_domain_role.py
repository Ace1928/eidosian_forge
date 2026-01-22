import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_create_domain_role(self):
    role_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'domain': self.project_domain_id}
    role = self.client.roles.create(**role_ref)
    self.addCleanup(self.client.roles.delete, role)
    self.check_role(role, role_ref)