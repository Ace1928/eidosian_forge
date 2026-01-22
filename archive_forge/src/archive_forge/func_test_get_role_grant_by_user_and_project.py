from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_get_role_grant_by_user_and_project(self):
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_bar['id'])
    self.assertEqual(1, len(roles_ref))
    PROVIDERS.assignment_api.create_grant(user_id=self.user_foo['id'], project_id=self.project_bar['id'], role_id=self.role_admin['id'])
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_bar['id'])
    self.assertIn(self.role_admin['id'], [role_ref['id'] for role_ref in roles_ref])
    PROVIDERS.assignment_api.create_grant(user_id=self.user_foo['id'], project_id=self.project_bar['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_bar['id'])
    roles_ref_ids = []
    for ref in roles_ref:
        roles_ref_ids.append(ref['id'])
    self.assertIn(self.role_admin['id'], roles_ref_ids)
    self.assertIn(default_fixtures.MEMBER_ROLE_ID, roles_ref_ids)