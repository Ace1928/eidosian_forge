import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def test_list_users_with_filters(self):
    suffix = uuid.uuid4().hex
    user1_ref = {'name': 'test_user' + suffix, 'domain': self.project_domain_id, 'default_project': self.project_id, 'password': uuid.uuid4().hex, 'description': uuid.uuid4().hex}
    user2_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'domain': self.project_domain_id, 'default_project': self.project_id, 'password': uuid.uuid4().hex, 'description': uuid.uuid4().hex}
    user1 = self.client.users.create(**user1_ref)
    self.client.users.create(**user2_ref)
    users = self.client.users.list(name__contains=['test_user', suffix])
    self.assertEqual(1, len(users))
    self.assertIn(user1, users)