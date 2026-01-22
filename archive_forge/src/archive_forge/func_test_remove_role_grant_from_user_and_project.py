from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
def test_remove_role_grant_from_user_and_project(self):
    PROVIDERS.assignment_api.create_grant(user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_baz['id'])
    self.assertDictEqual(self.role_member, roles_ref[0])
    PROVIDERS.assignment_api.delete_grant(user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)
    roles_ref = PROVIDERS.assignment_api.list_grants(user_id=self.user_foo['id'], project_id=self.project_baz['id'])
    self.assertEqual(0, len(roles_ref))
    self.assertRaises(exception.RoleAssignmentNotFound, PROVIDERS.assignment_api.delete_grant, user_id=self.user_foo['id'], project_id=self.project_baz['id'], role_id=default_fixtures.MEMBER_ROLE_ID)