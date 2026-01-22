import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_user_grouping(self):
    user = fixtures.User(self.client, self.project_domain_id)
    group = fixtures.Group(self.client, self.project_domain_id)
    self.useFixture(user)
    self.useFixture(group)
    self.assertRaises(http.NotFound, self.client.users.check_in_group, user.id, group.id)
    self.client.users.add_to_group(user.id, group.id)
    self.client.users.check_in_group(user.id, group.id)
    self.client.users.remove_from_group(user.id, group.id)
    self.assertRaises(http.NotFound, self.client.users.check_in_group, user.id, group.id)