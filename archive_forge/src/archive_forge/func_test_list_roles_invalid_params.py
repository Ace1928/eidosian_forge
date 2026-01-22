import uuid
from keystoneauth1.exceptions import http
from keystoneclient import exceptions
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_list_roles_invalid_params(self):
    user = fixtures.User(self.client, self.project_domain_id)
    self.useFixture(user)
    self.assertRaises(exceptions.ValidationError, self.client.roles.list, user=user.id)
    group = fixtures.Group(self.client, self.project_domain_id)
    self.useFixture(group)
    self.assertRaises(exceptions.ValidationError, self.client.roles.list, group=group.id)